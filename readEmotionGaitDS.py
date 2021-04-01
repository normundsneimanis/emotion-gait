import os
import h5py
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
import random
import argparse
import numpy as np
import math
from scipy.ndimage.interpolation import rotate
from scipy.spatial.transform import Rotation as R

emotionLabels = ('Angry', 'Neutral', 'Happy', 'Sad')

pn = {
    "root": 0,
    "spine": 1,
    "neck": 2,
    "head": 3,
    "left shoulder": 4,
    "left elbow": 5,
    "left hand": 6,
    "right shoulder": 7,
    "right elbow": 8,
    "right hand": 9,
    "left hip": 10,
    "left knee": 11,
    "left foot": 12,
    "right hip": 13,
    "right knee": 14,
    "right foot": 15,
}


class Point:
    def __init__(self, li):
        self.x = li[0]
        self.y = li[1]
        self.z = li[2]

    def li(self):
        return [self.x, self.y, self.z]

    def dist(self, p):
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2 + (self.z - p.z)**2)

    def rotate(self, deg, axis='y'):
        r = R.from_euler(axis, deg, degrees=True)
        self.x, self.y, self.z = np.matmul(r.as_matrix(), (self.x, self.y, self.z))
        pass

    def scale(self, amount):
        self.x *= amount
        self.y *= amount
        self.z *= amount



# Return array of coordinates for connection from, to
def connection(f, t, points):
    return [points[pn[f]].li(), points[pn[t]].li()]


def openFile(file):
    try:
        inputFile = h5py.File(file, 'r')
    except OSError as e:
        print("Failed to open file %s: %s" % (file, e.strerror))
        sys.exit(1)
    return inputFile


def updateGait(num, entryFrames, ax, lines, xmin, ymin, zmin, xmax, ymax, zmax):
    connections = [
        connection('root', 'spine', entryFrames[num]),
        connection('spine', 'neck', entryFrames[num]),
        connection('neck', 'head', entryFrames[num]),
        connection('neck', 'left shoulder', entryFrames[num]),
        connection('neck', 'right shoulder', entryFrames[num]),
        connection('left shoulder', 'left elbow', entryFrames[num]),
        connection('left elbow', 'left hand', entryFrames[num]),
        connection('right shoulder', 'right elbow', entryFrames[num]),
        connection('right elbow', 'right hand', entryFrames[num]),
        connection('root', 'left hip', entryFrames[num]),
        connection('root', 'right hip', entryFrames[num]),
        connection('left hip', 'left knee', entryFrames[num]),
        connection('right hip', 'right knee', entryFrames[num]),
        connection('right knee', 'right foot', entryFrames[num]),
        connection('left knee', 'left foot', entryFrames[num]),
    ]

    lines = Line3DCollection(connections, linewidths=1)
    ax.cla()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # front ax.view_init(azim=90, elev=-90)
    #ax.view_init(azim=69, elev=-69)
    ax.add_collection3d(lines)


def saveNewDataSet(datasetFile, gait, labelsFile=False, labels={}):
    hf = h5py.File(datasetFile, 'w')
    for gaitEntry in gait.keys():
        hf.create_dataset(gaitEntry, data=gait[gaitEntry], compression="gzip")
    hf.close()
    print("Saved %d entries to %s" % (len(gait), datasetFile))
    if labelsFile:
        hf = h5py.File(labelsFile, 'w')
        for gaitEntry in labels.keys():
            hf.create_dataset(gaitEntry, data=labels[gaitEntry])
        hf.close()
        print("Saved %d entries to %s" % (len(labels), labelsFile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View emotion-gait dataset.')
    parser.add_argument('-f', '--features', required=True,
                        help='Gait feature file path')
    parser.add_argument('-l', '--labels', required=False,
                        help='Labels file path, optional')
    parser.add_argument('--save-dir', required=False,
                        help='Save gait in given directory')
    parser.add_argument('--view-gait-num', required=False,
                        help='View gait with specified number. Random is used if not specified')
    parser.add_argument('--outliers', required=False, nargs=4, metavar=('joint', 'mean', 'std', 'stdnum'),
                        help='Outliers information'),
    parser.add_argument('--print-stats', required=False, action='store_true',
                        help='Print statistics'),
    parser.add_argument('--rotatex', required=False, type=int,
                        help='Rotate along x axes by num degrees')
    parser.add_argument('--rotatey', required=False, type=int,
                        help='Rotate along y axes by num degrees')
    parser.add_argument('--rotatez', required=False, type=int,
                        help='Rotate along z axes by num degrees')
    parser.add_argument('--scale', required=False, type=float,
                        help='Scale figure by this number')

    args = parser.parse_args()

    inputGaitFile = openFile(args.features)
    if args.labels:
        inputLabelFile = openFile(args.labels)

    gaitEntryNum = -1

    if args.outliers:
        joint, mean, std, stdnum = args.outliers
        joint = int(joint)
        mean = float(mean)
        stdnum = float(stdnum)
        std = float(std)
        print(joint, mean, std, stdnum)

    newDataSet = {}
    newDataSetLabels = {}

    while 1:
        if args.view_gait_num:
            if int(args.view_gait_num) > len(inputGaitFile.keys())-1:
                print("Requested gait number does not exist in dataset. Try smaller number")
                sys.exit(1)
            gaitEntryNum = int(args.view_gait_num)
        elif args.save_dir:
            if args.outliers:
                gaitEntryNum += 1
            else:
                #gaitEntryNum = random.randint(0, len(inputGaitFile.keys())-1)
                gaitEntryNum += 1
        else:
            if args.print_stats or args.outliers:
                gaitEntryNum += 1
            else:
                gaitEntryNum = random.randint(0, len(inputGaitFile.keys())-1)
        # Get the data
        if gaitEntryNum > len(inputGaitFile.keys()) - 1:
            print("End of dataset")
            break
        gaitEntry = list(inputGaitFile.keys())[gaitEntryNum]
        data = list(inputGaitFile[gaitEntry])
        if args.save_dir:
            if args.labels:
                saveFile = os.path.join(args.save_dir, ("%s-%s.html" % (gaitEntry, emotionLabels[inputLabelFile[gaitEntry][()]])))
                if os.path.exists(saveFile):
                    print("Skipping file %d: %s, exists" % (gaitEntryNum, gaitEntry))
                    continue
            else:
                saveFile = os.path.join(args.save_dir, ("%s.html" % (gaitEntry)))
                if os.path.exists(saveFile):
                    print("Skipping file %d: %s, exists" % (gaitEntryNum, gaitEntry))
                    continue
            # Not needed if saving modified dataset
            #if not args.outliers:
            #    with open(saveFile, 'w') as f:
            #        f.write(' ')

        if args.labels:
            print("Entry %d: %s consists of %d frames. Label: %s (%d)" %
                  (gaitEntryNum, gaitEntry, len(data),
                   emotionLabels[inputLabelFile[gaitEntry][()]], inputLabelFile[gaitEntry][()]))
        else:
            print("Entry %d: %s consists of %d frames." % (gaitEntryNum, gaitEntry, len(data)))
        entryFrames = []
        zmin = xmin = ymin = 999  # To determine frame limits
        zmax = xmax = ymax = -999
        prevFramePoints = []
        outlier = 0
        zeroMoveCount = 0

        for _ in range(7):  # Make 3 rotations for each gait
            rotateAmount = random.randint(45, 325)
            scaleAmount = random.randrange(50, 150) / 100
            newEntryName = gaitEntry + "-y-" + str(rotateAmount) + "-scale-" + str(scaleAmount)
            newDataSet[newEntryName] = []
            newDataSetLabels[newEntryName] = inputLabelFile[gaitEntry][()]
            print(newEntryName)

            dists = []  # Distances between movements. Used to statistically analyze dataset entries
            for p in range(16):
                dists.append([])

            skippedFramesCount = 0
            for frame in list(data):
                #print("Frame contains %d data entries" % len(frame))
                points = []
                diffs = [0, 0, 0]
                for prt in range(0, 16):
                    # Center the body around spine joint
                    if not args.print_stats and not args.outliers:
                        if prt == 0:
                            if not any(np.isclose(list(frame)[0:3], [1e-10, 1e-10, 1e-10])):
                                diffs = np.subtract([0, 0, 0], list(frame)[0:3])

                    point = Point(np.add(list(frame)[prt * 3: prt * 3 + 3], diffs))
                    point.rotate(rotateAmount, 'y')
                    point.scale(scaleAmount)
                    if args.rotatex:
                        point.rotate(args.rotatex, 'x')
                    if args.rotatey:
                        point.rotate(args.rotatey, 'y')
                    if args.rotatez:
                        point.rotate(args.rotatez, 'z')
                    if args.scale:
                        point.scale(args.scale)
                    points.append(point)

                    # Find frame limits so movement does not go out of border
                    if point.x > xmax:
                        xmax = point.x
                    if point.x < xmin:
                        xmin = point.x
                    if point.y > ymax:
                        ymax = point.y
                    if point.y < ymin:
                        ymin = point.y
                    if point.z > zmax:
                        zmax = point.z
                    if point.z < zmin:
                        zmin = point.z

                # Prevent "fat" figure in case movement is in small area
                if xmin > -0.6:
                    xmin = -0.6
                if xmax < 0.6:
                    xmax = 0.6
                if ymin > -0.6:
                    ymin = -0.6
                if ymax < 0.6:
                    ymax = 0.6
                if zmin > -0.6:
                    zmin = -0.6
                if zmax < 0.6:
                    zmax = 0.6

                # Drop frames with no movement
                if prevFramePoints:
                    zeroMoveCount2 = 0
                    for j in range(16):
                        if points[j].dist(prevFramePoints[j]) < 1e-10:
                            zeroMoveCount2 += 1

                    if zeroMoveCount2 == 16:
                        skippedFramesCount += 1
                        continue

                entryFrames.append(points)

                if args.print_stats or args.outliers:
                    if not prevFramePoints:
                        prevFramePoints = points
                        continue
                    for p in range(len(points)):
                        dists[p].append(points[p].dist(prevFramePoints[p]))
                    if args.outliers:
                        if (points[joint].dist(prevFramePoints[joint]) < mean - stdnum * std or
                                points[joint].dist(prevFramePoints[joint]) > mean + stdnum * std) and not outlier:
                            print("Detected outlier")
                            outlier = 1
                        # Catch zero move
                        if np.isclose([points[joint].dist(prevFramePoints[joint])], [1e-10]):
                            zeroMoveCount += 1
                            if joint == 0.0 and zeroMoveCount >= 200 and not outlier:
                                print("Detected outlier from zero move count (root)")
                                outlier = 1
                            if (joint == 6.0 or joint == 9.0) and zeroMoveCount >= 200 and not outlier:
                                print("Detected outlier from zero move count (hands)")
                                outlier = 1

                prevFramePoints = points

                modifiedFrames = []
                for p in points:
                    modifiedFrames.append(p.x)
                    modifiedFrames.append(p.y)
                    modifiedFrames.append(p.z)
                newDataSet[newEntryName].append(modifiedFrames)

            if skippedFramesCount:
                print("Skipped %d stopped frames; %d total" % (skippedFramesCount, len(newDataSet[newEntryName])))

        # if gaitEntryNum > 5:
        #     break


    # TODO Make new dataset with centered bodies and add randomly rotated bodies
    # saveNewDataSet(os.path.join('emotion-gait', 'features_ELMD_centered.h5'), newDataSet)  # Centered
    # saveNewDataSet(os.path.join('emotion-gait', 'features_ELMD_rotated-y.h5'), newDataSet,
    #               os.path.join('emotion-gait', 'labels_ELMD_rotated-y.h5'), newDataSetLabels)  # Rotated ELMD
    # saveNewDataSet(os.path.join('emotion-gait', 'features_ELMD_rotated-y-scaled.h5'), newDataSet,
    #               os.path.join('emotion-gait', 'labels_ELMD_rotated-y-scaled.h5'), newDataSetLabels)  # Rotated + scaled ELMD
    # saveNewDataSet(os.path.join('emotion-gait', 'features_rotated-y.h5'), newDataSet,
    #               os.path.join('emotion-gait', 'labels_rotated-y.h5'), newDataSetLabels)  # Rotated
    saveNewDataSet(os.path.join('emotion-gait', 'features_rotated-y-scaled.h5'), newDataSet,
                  os.path.join('emotion-gait', 'labels_rotated-y-scaled.h5'), newDataSetLabels)  # Rotated


    #     if (not args.print_stats and not args.outliers) or outlier:
    #         fig = plt.figure(figsize=(1024 / 96, 1024 / 96), dpi=96)
    #         ax = fig.add_subplot(111, projection='3d', xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
    #         ax.view_init(azim=69, elev=-69)
    #
    #         connections = [
    #             connection('root', 'spine', entryFrames[0]),
    #             connection('spine', 'neck', entryFrames[0]),
    #             connection('neck', 'head', entryFrames[0]),
    #             connection('neck', 'left shoulder', entryFrames[0]),
    #             connection('neck', 'right shoulder', entryFrames[0]),
    #             connection('left shoulder', 'left elbow', entryFrames[0]),
    #             connection('left elbow', 'left hand', entryFrames[0]),
    #             connection('right shoulder', 'right elbow', entryFrames[0]),
    #             connection('right elbow', 'right hand', entryFrames[0]),
    #             connection('root', 'left hip', entryFrames[0]),
    #             connection('root', 'right hip', entryFrames[0]),
    #             connection('left hip', 'left knee', entryFrames[0]),
    #             connection('right hip', 'right knee', entryFrames[0]),
    #             connection('right knee', 'right foot', entryFrames[0]),
    #             connection('left knee', 'left foot', entryFrames[0]),
    #         ]
    #
    #         #lines, = ax.plot(pointsX, pointsY, pointsZ, 'o-', lw=2)
    #         lines = Line3DCollection(connections, linewidths=1)
    #         #name_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    #         #frame_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    #         ax.add_collection3d(lines)
    #         ax.set_title(gaitEntry)
    #
    #         line_ani = animation.FuncAnimation(fig,
    #                                            updateGait,
    #                                            len(data),
    #                                            fargs=(entryFrames, ax, lines, xmin, ymin, zmin, xmax, ymax, zmax),
    #                                            interval=50, blit=False)
    #
    #         if args.save_dir:
    #             if not os.path.isfile(saveFile) or not args.outliers:
    #                 with open(saveFile, "w") as f:
    #                     print(line_ani.to_html5_video(), file=f)
    #                 print("Saved %s." % gaitEntry)
    #                 plt.close()
    #             else:
    #                 print("File for %s exists. Continuing." % gaitEntry)
    #         else:
    #             plt.show()
    #             if not args.outliers:
    #                 sys.exit()
    #
    #         if args.view_gait_num:
    #             sys.exit()
    #
    # if args.print_stats:
    #     for p in range(16):
    #         print("%d: avg: %.6f median: %.6f stddev: %.6f, max: %.3f" %
    #               (p, np.mean(dists[p]), np.median(dists[p]), np.std(dists[p]), np.max(dists[p])))
