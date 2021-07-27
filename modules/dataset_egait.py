import h5py
import sys
import os
import requests
import numpy as np
import torch
from tqdm import tqdm
import datetime
import math
import random
# from scipy.ndimage.interpolation import rotate
from scipy.spatial.transform import Rotation as R
from modules.visualize_gait import *
import time
from modules.compute_aff_features.compute_features import compute_features
from modules.compute_aff_features.normalize_features import normalize_features
from modules.compute_aff_features.cross_validate import cross_validate


class DatasetEGait(torch.utils.data.Dataset):
    save_videos = int(os.getenv("SAVE_VIDEOS", "0"))  # For augmentation comparison video creation
    max_save_videos = int(os.getenv("MAX_SAVE_VIDEOS", "100"))  # What is maximum number of videos to save
    saveGaitName = os.getenv("SAVE_GAIT_NAME", "")
    loaded = 0
    gaits = np.array([])
    lengths = np.array([])
    labels = np.array([])
    train_gaits = []
    test_gaits = []
    path = 'emotion-gait'
    labelsCount = []
    startedFile = 'emotion-gait.started'
    skiplist_ELMD = ["000412", "000413", "000422", "000423", "000734", "001367", "001590", "001666",  # jumps
                     "000348", "000403", "000412", "000413", "001590", "001707", "001626", "001643",
                     "001666"  # no movement
                     ]

    def __init__(self, is_train, dataset_files,
                 remove_zeromove=False,
                 center_root=False,
                 rotate_y=False,
                 scale=False,
                 drop_elmd_frames=False,
                 normalize_gait_sizes=False,
                 memmap=False,
                 memmap_only=False,
                 affective_features=False):
        self.is_train = is_train
        self.datasetFiles = []
        for i in range(len(dataset_files[0])):
            if len(dataset_files[1]):
                self.datasetFiles.append([dataset_files[0][i], dataset_files[1][i]])
            else:
                self.datasetFiles.append([dataset_files[0][i], None])

        self.remove_zeromove = remove_zeromove
        self.center_root = center_root
        self.rotate_y = rotate_y
        self.scale = scale
        self.drop_elmd_frames = drop_elmd_frames
        self.normalize_gait_sizes = normalize_gait_sizes
        self.memmap = memmap
        self.memmap_only = memmap_only
        self.affective_features = affective_features

        if self.memmap_only:
            self.memmap = True

        if not DatasetEGait.loaded:
            if self.memmap and os.path.isfile(os.path.join(self.findMmapDir(), self.startedFile)) and not self.memmap_only:
                self.loadMemmapDataSet()
                self.splitDataSet()
            else:
                self.loadDataSet()
                self.splitDataSet()

        if is_train:
            self.gaitIds = DatasetEGait.train_gaits
        else:
            self.gaitIds = DatasetEGait.test_gaits

    def loadMemmapDataSet(self):
        mmapFileDone = os.path.join(self.findMmapDir(), "emotion-gait.done")
        numChecks = 0
        if not os.path.isfile(mmapFileDone):
            print("Waiting for mmap file to be done ...")
        while not os.path.isfile(mmapFileDone):
            numChecks += 1
            if numChecks > 300:
                print("Wait time for %s exceeded" % mmapFileDone)
                sys.exit(2)
            time.sleep(1)

        totalGaits = self.getNumGaits()

        mmapFileGaits = os.path.join(self.findMmapDir(), "emotion-gait.gaits.mmap")
        DatasetEGait.gaits = np.memmap(
            mmapFileGaits,
            mode='r+',
            dtype=np.float32,
            shape=(totalGaits, 240, 48)
        )
        mmapFileLengths = os.path.join(self.findMmapDir(), "emotion-gait.lengths.mmap")
        DatasetEGait.lengths = np.memmap(
            mmapFileLengths,
            mode='r+',
            dtype=np.int64,
            shape=(totalGaits)
        )
        mmapFileLabels = os.path.join(self.findMmapDir(), "emotion-gait.labels.mmap")
        DatasetEGait.labels = np.memmap(
            mmapFileLabels,
            mode='r+',
            dtype=np.int64,
            shape=(totalGaits)
        )

        DatasetEGait.loaded = 1

    def getNumGaits(self):
        totalGaits = 0
        for file, labelsFile in self.datasetFiles:
            inputGaitFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, file))
            totalGaits += len(inputGaitFile.keys())
            if 'ELMD' in file:
                totalGaits -= len(self.skiplist_ELMD)
            inputGaitFile.close()
        self.totalGaits = totalGaits
        return self.totalGaits

    def loadDataSet(self):
        print("Loading emotion-gait dataset")
        startTime = datetime.datetime.utcnow()

        if self.memmap:
            with open(os.path.join(self.findMmapDir(), self.startedFile), 'w') as f:
                pass

        # Download dataset files if needed
        for file, labelsFile in self.datasetFiles:
            if not os.path.isfile(os.path.join(DatasetEGait.path, file)):
                DatasetEGait.downloadFile('http://nneimanis.id.lv/emotion-gait/' + file)
            if labelsFile and not os.path.isfile(os.path.join(DatasetEGait.path, labelsFile)):
                DatasetEGait.downloadFile('http://nneimanis.id.lv/emotion-gait/' + labelsFile)

        # Collect mean shoulder width and height
        meanHeight = 0
        if self.normalize_gait_sizes:
            # Get mean height and shoulder width of a gait for all
            print("Calculating mean body size")
            startTimeLoadSizes = datetime.datetime.utcnow()
            distsRoot_Neck = []
            for file, labelsFile in self.datasetFiles:
                inputGaitFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, file))
                for gaitName in inputGaitFile.keys():
                    frames = inputGaitFile[gaitName]
                    for frame in list(frames):
                        pointRoot = Point(list(frame)[0 * 3: 0 * 3 + 3])  # Root
                        pointNeck = Point(list(frame)[2 * 3: 2 * 3 + 3])  # Neck
                        distsRoot_Neck.append(pointRoot.dist(pointNeck))

            meanHeight = np.mean(distsRoot_Neck)
            timeDiffLoadSizes = (datetime.datetime.utcnow() - startTimeLoadSizes).total_seconds()
            print("mean spine height: %.2f, calculation done in %.1f seconds. " % (meanHeight, timeDiffLoadSizes))

        gaitNum = 0
        if self.memmap:
            # Count number of gaits so we can construct mmap file
            totalGaits = self.getNumGaits()

            mmapFileGaits = os.path.join(self.findMmapDir(), "emotion-gait.gaits.mmap")
            gaits = np.memmap(
                mmapFileGaits,
                mode='w+',
                dtype=np.float32,
                shape=(totalGaits, 240, 48)
            )
            mmapFileLengths = os.path.join(self.findMmapDir(), "emotion-gait.lengths.mmap")
            lengths = np.memmap(
                mmapFileLengths,
                mode='w+',
                dtype=np.int64,
                shape=(totalGaits)
            )
            mmapFileLabels = os.path.join(self.findMmapDir(), "emotion-gait.labels.mmap")
            labels = np.memmap(
                mmapFileLabels,
                mode='w+',
                dtype=np.int64,
                shape=(totalGaits)
            )
        else:
            gaits = []
            lengths = []
            labels = []

        if self.affective_features:
            features = []  # Affective features
            time_step = 1.0 / 30.0

        for file, labelsFile in self.datasetFiles:
            print("\tLoading %s" % file)
            labelsCountFile = [0, 0, 0, 0]
            inputGaitFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, file))
            inputLabelFile = None
            if labelsFile:
                inputLabelFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, labelsFile))

            prevFramePoints = []
            skippedFramesCount = 0
            if self.save_videos and not os.path.isdir("gait-videos"):
                os.mkdir("gait-videos")
            for gaitName in inputGaitFile.keys():
                if 'ELMD' in file:
                    if gaitName[:6] in self.skiplist_ELMD:
                        print("\t\tSkipping gait %s" % gaitName)
                        continue

                vis = VisualizeGait(return_video=True)

                if self.save_videos:
                    if (self.saveGaitName and self.saveGaitName in gaitName) or (not self.saveGaitName and gaitNum % 10 == 0):
                        if not os.path.isfile(os.path.join("gait-videos", f"{gaitName}-orig.html")):
                            with open(os.path.join("gait-videos", f"{gaitName}-orig.html"), "w") as f:
                                print(vis.vizualize(inputGaitFile[gaitName], gaitName), file=f)

                newFrames = []
                frames = inputGaitFile[gaitName]
                frameNum = -1
                normalizatonScale = 0
                for frame in list(frames):
                    frameNum += 1
                    if self.drop_elmd_frames and 'ELMD' in file:
                        if frameNum % 2 == 0:
                            continue

                    if self.normalize_gait_sizes and frameNum == 0:
                        pointRoot = Point(list(frame)[0 * 3: 0 * 3 + 3])  # Root
                        pointNeck = Point(list(frame)[2 * 3: 2 * 3 + 3])  # Neck
                        normalizatonScale = pointRoot.dist(pointNeck) / meanHeight
                        if normalizatonScale < 1.1 and normalizatonScale > 0.9:
                            normalizatonScale = 0

                    if not self.remove_zeromove and not self.center_root and not self.normalize_gait_sizes:
                        newFrames.append(frame)
                        continue

                    # Remove frames w/o movement
                    points = []
                    diffs = [0, 0, 0]
                    for prt in range(0, 16):
                        # Center root point
                        if self.center_root and prt == 0:
                            if not any(np.isclose(list(frame)[0:3], [1e-10, 1e-10, 1e-10])):
                                diffs = np.subtract([0, 0, 0], list(frame)[0:3])
                        point = Point(np.add(list(frame)[prt * 3: prt * 3 + 3], diffs))
                        if normalizatonScale:
                            point.scale(normalizatonScale)
                        points.append(point)

                    # Drop frames with no movement
                    if self.remove_zeromove and prevFramePoints:
                        zeroMoveCount2 = 0
                        for j in range(16):
                            if points[j].dist(prevFramePoints[j]) < 1e-10:
                                zeroMoveCount2 += 1

                        if zeroMoveCount2 == 16:
                            skippedFramesCount += 1
                            continue

                    prevFramePoints = points

                    modifiedFrames = []
                    for p in points:
                        modifiedFrames.append(p.x)
                        modifiedFrames.append(p.y)
                        modifiedFrames.append(p.z)
                    newFrames.append(modifiedFrames)

                frameLen = len(newFrames)
                # pad frames so we have 240 frame for each gait
                newFrames = np.pad(newFrames, ((0, 240 - len(newFrames)), (0, 0)))

                # Save gait augmentations for visualization
                if self.save_videos and (self.rotate_y or self.scale):
                    gait = []
                    if self.rotate_y:
                        rotateAmount = random.randint(45, 325)
                    else:
                        rotateAmount = 0
                    if self.scale:
                        scaleAmount = random.randrange(50, 150) / 100
                    else:
                        scaleAmount = 1.0
                    r = R.from_euler('y', rotateAmount, degrees=True).as_matrix().T

                    frames = np.copy(newFrames)
                    for frame in list(frames):
                        gait.append(np.matmul(frame.reshape(-1, 3), r).flatten() * scaleAmount)
                    newFrames = np.copy(gait)

                if self.save_videos:
                    if ((self.saveGaitName and self.saveGaitName in gaitName) or
                            (not self.saveGaitName and gaitNum % 10 == 0)):
                        fname = os.path.join("gait-videos", f"{gaitName}-%d-%d-%d-%d-%d.html" %
                                             (self.remove_zeromove, self.center_root, self.rotate_y,
                                              self.scale, self.normalize_gait_sizes))
                        if not os.path.isfile(fname):
                            with open(fname, "w") as f:
                                print(vis.vizualize(newFrames, gaitName), file=f)
                        # Exit when selected gait video has been saved
                        if self.saveGaitName:
                            sys.exit(0)
                if self.save_videos and gaitNum / 10 > self.max_save_videos:
                    sys.exit(0)

                # if normalizatonScale:
                #     print("normalizationScale: %.2f" % normalizatonScale)
                #     vis = VisualizeGait()
                #     vis.vizualize(newFrames, gaitName)

                if inputLabelFile:
                    label = inputLabelFile[gaitName][()]
                else:
                    if 'Angry' in gaitName:
                        label = 0
                    elif 'Happy' in gaitName:
                        label = 1
                    elif 'Neutral' in gaitName:
                        label = 2
                    elif 'Sad' in gaitName:
                        label = 3
                    else:
                        print("Error: unrecognized emotion label from gait name [%s]" % gaitName)

                if self.memmap:
                    if gaitNum == self.totalGaits:
                        print("Error: Gait num exceeded on gait: %s" % gaitName)
                        break
                    gaits[gaitNum] = np.array(newFrames).astype(np.float32)
                    lengths[gaitNum] = frameLen
                    labels[gaitNum] = label
                else:
                    gaits.append(newFrames)
                    lengths.append(frameLen)
                    labels.append(label)

                if self.affective_features:
                    frames = np.array(newFrames)
                    feature = []  # [label]
                    if frames.ndim == 2:
                        feature += compute_features(frames, time_step)
                    else:
                        feature += frames
                    features.append(feature)

                gaitNum += 1
                if self.affective_features and gaitNum != len(features):
                    print("Feature count does not match gait count")
                labelsCountFile[label] += 1
            print("Labels count for file " + file + ": " + str(labelsCountFile))

        if self.affective_features:
            normalized_features = []
            normalize_features(features, normalized_features)
            # print("Gaits: %d, Normalized features: %d, Features: %d" % (len(gaits), len(normalized_features), len(features)))
            # cross_validate(normalized_features, labels)
            for num in range(0, len(gaits)):
                if num == len(gaits) or num == len(normalized_features):
                    print("Affective feature number of gaits exceeded")
                if not self.memmap:
                    gaits[num].append(np.pad(np.nan_to_num(normalized_features[num]), (0, 48 - len(normalized_features[num]))))
                else:
                    # print(
                    #     "num: %d gait len: %d feature len: %d" % (num, len(gaits[num]), len(normalized_features[num])))
                    if lengths[num] < 240:
                        gaits[num][lengths[num]] = np.pad(np.nan_to_num(normalized_features[num]),
                                                 (0, 48 - len(normalized_features[num])))
                        lengths[num] = lengths[num] + 1
                    else:
                        gaits[num][239] = np.pad(np.nan_to_num(normalized_features[num]),
                                                 (0, 48 - len(normalized_features[num])))

        if self.memmap:
            DatasetEGait.gaits = gaits
            DatasetEGait.gaits.flush()
            DatasetEGait.lengths = lengths
            DatasetEGait.lengths.flush()
            DatasetEGait.labels = labels
            DatasetEGait.labels.flush()
            mmapFileDone = os.path.join(self.findMmapDir(), "emotion-gait.done")
            with open(mmapFileDone, 'w') as f:
                pass
        else:
            DatasetEGait.gaits = np.array(gaits).astype(np.float32)
            DatasetEGait.lengths = np.array(lengths).astype(np.int64)
            DatasetEGait.labels = np.array(labels).astype(np.int64)

        timeDiff = (datetime.datetime.utcnow() - startTime).total_seconds()
        print("Done in %.1f seconds." % timeDiff)

        DatasetEGait.loaded = 1
        if self.memmap_only:
            print("Memmap files created. Exiting.")
            sys.exit(0)

    def splitDataSet(self):
        # Split in test and train data
        print("Splitting in train and test parts")
        for i in range(len(DatasetEGait.gaits)):
            if i % 5 == 0:
                DatasetEGait.test_gaits.append(i)
            else:
                DatasetEGait.train_gaits.append(i)

        labelsCount = [0, 0, 0, 0]
        for l in DatasetEGait.labels:
            labelsCount[l] += 1

        print("Train len: %d. Test len: %d." %
              (len(DatasetEGait.train_gaits), len(DatasetEGait.test_gaits)))
        print("Labels count: " + str(labelsCount))
        DatasetEGait.labelsCount = labelsCount

    def findMmapDir(self):
        dirs = [os.path.join(os.sep, "scratch", "nneimanis"), os.path.join(os.sep, "tmp")]
        for d in dirs:
            if os.path.isdir(d):
                return d

        raise Exception("Temporary directory not found in list: " + str(dirs))

    def waitForMmap(self):
        file = os.path.join(elf.findMmapDir(), "emotion-gait.gaits.mmap.done")
        while not os.path.isfile(file):
            time.sleep(1)

    def __len__(self):
        return len(self.gaitIds)

    def __getitem__(self, idx):
        # Scale making features "larger" or "smaller"
        gaitId = self.gaitIds[idx]
        if self.rotate_y or self.scale:
            gait = []
            if self.rotate_y:
                rotateAmount = random.randint(45, 325)
            else:
                rotateAmount = 0
            if self.scale:
                scaleAmount = random.randrange(50, 150) / 100
            else:
                scaleAmount = 1.0
            r = R.from_euler('y', rotateAmount, degrees=True).as_matrix().T

            frames = np.copy(DatasetEGait.gaits[gaitId])
            for frame in list(frames):
                gait.append(np.matmul(frame.reshape(-1, 3), r).flatten() * scaleAmount)
            return np.array(gait).astype(np.float32), self.lengths[gaitId], self.labels[gaitId]
        else:
            return self.gaits[gaitId], self.lengths[gaitId], self.labels[gaitId]

    @staticmethod
    def openFile(file):
        try:
            inputFile = h5py.File(file, 'r')
        except OSError as e:
            print("Failed to open file %s: %s" % (file, e.strerror))
            sys.exit(1)
        return inputFile

    @staticmethod
    def downloadFile(url):
        local_filename = url.split('/')[-1]
        if not os.path.isdir(DatasetEGait.path):
            os.mkdir(DatasetEGait.path)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_length = int(r.headers.get('content-length'))
            with open(os.path.join(DatasetEGait.path, local_filename), 'wb') as f:
                # for chunk in r.iter_content(chunk_size=8192):
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=int(total_length/8192), unit=' chunks'):
                    f.write(chunk)
        return local_filename
