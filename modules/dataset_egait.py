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


class DatasetEGait(torch.utils.data.Dataset):
    loaded = 0
    gaits = np.array([])
    lengths = np.array([])
    labels = np.array([])
    train_gaits = []
    test_gaits = []
    path = 'emotion-gait'
    labelsCount = []

    def __init__(self, is_train, dataset_files,
                 remove_zeromove=False,
                 center_root=False,
                 rotate_y=False,
                 scale=False,
                 equalize=False,
                 drop_elmd_frames=False,
                 normalize_gait_sizes=False):
        self.is_train = is_train
        self.datasetFiles = []
        for i in range(len(dataset_files[0])):
            self.datasetFiles.append([dataset_files[0][i], dataset_files[1][i]])

        self.remove_zeromove = remove_zeromove
        self.center_root = center_root
        self.rotate_y = rotate_y
        self.scale = scale
        self.equalize = equalize
        self.drop_elmd_frames = drop_elmd_frames
        self.normalize_gait_sizes = normalize_gait_sizes

        if not DatasetEGait.loaded:
            self.loadDataSet()

        if is_train:
            self.gaitIds = DatasetEGait.train_gaits
        else:
            self.gaitIds = DatasetEGait.test_gaits

    def loadDataSet(self):
        print("Loading emotion-gait dataset")
        startTime = datetime.datetime.utcnow()
        skiplist_ELMD = ["000412", "000413", "000422", "000423", "000734", "001367", "001590", "001666",  # jumps
                         "000348", "000403", "000412", "000413", "001590", "001707", "001626", "001643",
                         "001666"  # no movement
                         ]
        gaits = []
        lengths = []
        labels = []
        labelsCount = [0, 0, 0, 0]
        labelsEmotionsIds = [[], [], [], []]

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


        for file, labelsFile in self.datasetFiles:
            print("\tLoading %s" % file)
            labelsCountFile = [0, 0, 0, 0]
            inputGaitFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, file))
            inputLabelFile = None
            if labelsFile:
                inputLabelFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, labelsFile))

            prevFramePoints = []
            skippedFramesCount = 0
            for gaitName in inputGaitFile.keys():
                if 'ELMD' in file:
                    if gaitName[:6] in skiplist_ELMD:
                        print("\t\tSkipping gait %s" % gaitName)
                        continue

                newFrames = []
                frames = inputGaitFile[gaitName]
                frameNum = -1
                normalizatonScale = 0
                for frame in list(frames):
                    frameNum += 1
                    if 'ELMD' in file and self.drop_elmd_frames:
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

                # zero pad frames so we have 240 frame for each gait
                frameLen = len(newFrames)
                while len(newFrames) < 240:
                    newFrames.append(newFrames[len(newFrames)-1])

                # if normalizatonScale:
                #     print("normalizationScale: %.2f" % normalizatonScale)
                #     vis = VisualizeGait()
                #     vis.vizualize(newFrames, gaitName)

                gaits.append(newFrames)
                lengths.append(frameLen)
                labels.append(inputLabelFile[gaitName][()])
                labelsCount[inputLabelFile[gaitName][()]] += 1
                labelsCountFile[inputLabelFile[gaitName][()]] += 1
                labelsEmotionsIds[inputLabelFile[gaitName][()]].append(len(lengths) - 1)
            print("Labels count for file " + file + ": " + str(labelsCountFile))

        if self.equalize:
            len1 = len(labelsEmotionsIds[1]) - 1
            while len(labelsEmotionsIds[0]) > len(labelsEmotionsIds[1]):
                rand = random.randint(0, len1)
                gaits.append(gaits[labelsEmotionsIds[1][rand]])
                lengths.append(lengths[labelsEmotionsIds[1][rand]])
                labels.append(labels[labelsEmotionsIds[1][rand]])
                labelsEmotionsIds[1].append(labels[labelsEmotionsIds[1][rand]])
            len2 = len(labelsEmotionsIds[2]) - 1
            while len(labelsEmotionsIds[0]) > len(labelsEmotionsIds[2]):
                rand = random.randint(0, len2)
                gaits.append(gaits[labelsEmotionsIds[2][rand]])
                lengths.append(lengths[labelsEmotionsIds[2][rand]])
                labels.append(labels[labelsEmotionsIds[2][rand]])
                labelsEmotionsIds[2].append(labels[labelsEmotionsIds[2][rand]])
            len3 = len(labelsEmotionsIds[3]) - 1
            while len(labelsEmotionsIds[0]) > len(labelsEmotionsIds[3]):
                rand = random.randint(0, len3)
                gaits.append(gaits[labelsEmotionsIds[3][rand]])
                lengths.append(lengths[labelsEmotionsIds[3][rand]])
                labels.append(labels[labelsEmotionsIds[3][rand]])
                labelsEmotionsIds[3].append(labels[labelsEmotionsIds[3][rand]])

        DatasetEGait.gaits = np.array(gaits).astype(np.float32)
        DatasetEGait.lengths = np.array(lengths).astype(np.int64)
        DatasetEGait.labels = np.array(labels).astype(np.int64)

        # Split in test and train data
        print("Splitting in train and test parts")
        for i in range(len(DatasetEGait.gaits)):
            if i % 5 == 0:
                DatasetEGait.test_gaits.append(i)
            else:
                DatasetEGait.train_gaits.append(i)

        timeDiff = (datetime.datetime.utcnow() - startTime).total_seconds()
        print("Done in %.1f seconds. Train len: %d. Test len: %d." % (timeDiff, len(DatasetEGait.train_gaits), len(DatasetEGait.test_gaits)))
        print("Labels count: " + str(labelsCount))
        DatasetEGait.labelsCount = labelsCount
        if self.equalize:
            labelsCount = [0, 0, 0, 0]
            for i in range(len(labels)):
                labelsCount[labels[i]] += 1
            print("Labels count after eq: " + str(labelsCount))

        DatasetEGait.loaded = 1

    def __len__(self):
        return len(self.gaitIds)

    def __getitem__(self, idx):
        # Scale making features "larger" or "smaller"
        gaitId = self.gaitIds[idx]
        if self.rotate_y or self.scale:
            gait = []
            rotateAmount = random.randint(45, 325)
            scaleAmount = random.randrange(50, 150) / 100
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
