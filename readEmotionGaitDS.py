import os
import h5py
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
import random
import argparse

emotionLabels = ("happy", "angry", "neutral", "sad")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View emotion-gait dataset.')
    parser.add_argument('-f', '--file-gait', required=True,
                        help='Gait file path')
    parser.add_argument('--file-features', required=False,
                        help='Feature file path')
    parser.add_argument('--save-dir', required=False,
                        help='Save gait in given directory')
    parser.add_argument('--view-gait-num', required=False,
                        help='View gait with specified number. Random is used if not specified')

    args = parser.parse_args()

    inputGaitFile = openFile(args.file_gait)
    if args.file_features:
        inputLabelFile = openFile(args.file_features)

    gaitEntryNum = -1
    while 1:
        if args.view_gait_num:
            if int(args.view_gait_num) > len(inputGaitFile.keys())-1:
                print("Requested gait number does not exist in dataset. Try smaller number")
                sys.exit(1)
            gaitEntryNum = int(args.view_gait_num)
        elif args.save_dir:
            gaitEntryNum = gaitEntryNum + 1
        else:
            gaitEntryNum = random.randint(0, len(inputGaitFile.keys())-1)
        # Get the data
        if gaitEntryNum > len(inputGaitFile.keys()) - 1:
            print("End of dataset")
            sys.exit()
        gaitEntry = list(inputGaitFile.keys())[gaitEntryNum]
        data = list(inputGaitFile[gaitEntry])
        if args.save_dir:
            if args.file_features:
                saveFile = os.path.join(args.save_dir, ("%s-%d.html" % (gaitEntry, inputLabelFile[gaitEntry][()])))
                if os.path.exists(saveFile):
                    print("Skipping file %d: %s, exists" % (gaitEntryNum, gaitEntry))
                    continue
            else:
                saveFile = os.path.join(args.save_dir, ("%s.html" % (gaitEntry)))
                if os.path.exists(saveFile):
                    print("Skipping file %d: %s, exists" % (gaitEntryNum, gaitEntry))
                    continue
            with open(saveFile, 'w') as f:
                f.write(' ')

        if args.file_features:
            print("Entry %d: %s consists of %d frames. Label: %s (%d)" % (gaitEntryNum, gaitEntry, len(data), emotionLabels[inputLabelFile[gaitEntry][()]], inputLabelFile[gaitEntry][()]))
        else:
            print("Entry %d: %s consists of %d frames." % (gaitEntryNum, gaitEntry, len(data)))
        entryFrames = []
        zmin = xmin = ymin = 999  # To determine frame limits
        zmax = xmax = ymax = -999
        for frame in list(data):
            #print("Frame contains %d data entries" % len(frame))
            points = []
            for prt in range(0, 16):
                if list(frame)[prt * 3] > xmax:
                    xmax = list(frame)[prt * 3]
                if list(frame)[prt * 3] < xmin:
                    xmin = list(frame)[prt * 3]
                if list(frame)[prt * 3 + 1] > ymax:
                    ymax = list(frame)[prt * 3 + 1]
                if list(frame)[prt * 3 + 1] < ymin:
                    ymin = list(frame)[prt * 3 + 1]
                if list(frame)[prt * 3 + 2] > zmax:
                    zmax = list(frame)[prt * 3 + 2]
                if list(frame)[prt * 3 + 2] < zmin:
                    zmin = list(frame)[prt * 3 + 2]
                points.append(Point(list(frame)[prt * 3: prt * 3 + 3]))

            entryFrames.append(points)

        fig = plt.figure(figsize=(1024 / 96, 1024 / 96), dpi=96)
        ax = fig.add_subplot(111, projection='3d', xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
        ax.view_init(azim=69, elev=-69)

        connections = [
            connection('root', 'spine', entryFrames[0]),
            connection('spine', 'neck', entryFrames[0]),
            connection('neck', 'head', entryFrames[0]),
            connection('neck', 'left shoulder', entryFrames[0]),
            connection('neck', 'right shoulder', entryFrames[0]),
            connection('left shoulder', 'left elbow', entryFrames[0]),
            connection('left elbow', 'left hand', entryFrames[0]),
            connection('right shoulder', 'right elbow', entryFrames[0]),
            connection('right elbow', 'right hand', entryFrames[0]),
            connection('root', 'left hip', entryFrames[0]),
            connection('root', 'right hip', entryFrames[0]),
            connection('left hip', 'left knee', entryFrames[0]),
            connection('right hip', 'right knee', entryFrames[0]),
            connection('right knee', 'right foot', entryFrames[0]),
            connection('left knee', 'left foot', entryFrames[0]),
        ]

        #lines, = ax.plot(pointsX, pointsY, pointsZ, 'o-', lw=2)
        lines = Line3DCollection(connections, linewidths=1)
        #name_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        #frame_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        ax.add_collection3d(lines)
        ax.set_title(gaitEntry)

        line_ani = animation.FuncAnimation(fig,
                                           updateGait,
                                           len(data),
                                           fargs=(entryFrames, ax, lines, xmin, ymin, zmin, xmax, ymax, zmax),
                                           interval=50, blit=False)

        if args.save_dir:
            with open(saveFile, "w") as f:
                print(line_ani.to_html5_video(), file=f)
            print("Saved %s." % gaitEntry)
            plt.close()
        else:
            plt.show()
            sys.exit()

        if args.view_gait_num:
            sys.exit()
