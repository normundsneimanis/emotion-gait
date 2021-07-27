import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


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

    def scale(self, amount):
        self.x *= amount
        self.y *= amount
        self.z *= amount


class VisualizeGait:
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

    def __init__(self, return_video=False):
        self.returnVideo = return_video
        pass

    @staticmethod
    def fig2data(fig):
        # function to convert matplotlib fig to numpy
        # draw the renderer
        fig.canvas.draw()

        # Get the RGB buffer from the figure
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def connection(f, t, points):
        # Return array of coordinates for connection from, to
        return [points[VisualizeGait.pn[f]].li(), points[VisualizeGait.pn[t]].li()]

    @staticmethod
    def getConnections(entryFrames, num):
        connection = VisualizeGait.connection
        return [
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

    @staticmethod
    def updateGait(num, entryFrames, ax, lines, xmin, ymin, zmin, xmax, ymax, zmax):
        connections = VisualizeGait.getConnections(entryFrames, num)

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

    def vizualize(self, gait_data, gait_name):
        zmin = xmin = ymin = 999  # To determine frame limits
        zmax = xmax = ymax = -999
        entryFrames = []
        for frame in gait_data:
            points = []
            for prt in range(0, 16):
                point = Point(list(frame)[prt * 3: prt * 3 + 3])
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
            entryFrames.append(points)

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

        plt.close()
        # fig = plt.figure(figsize=(1024 / 96, 1024 / 96), dpi=96)
        fig = plt.figure(figsize=(200 / 96, 200 / 96), dpi=96)
        ax = fig.add_subplot(111, projection='3d', xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
        ax.view_init(azim=69, elev=-69)

        connections = VisualizeGait.getConnections(entryFrames, 0)

        lines = Line3DCollection(connections, linewidths=1)
        ax.add_collection3d(lines)
        ax.set_title(gait_name)

        line_ani = animation.FuncAnimation(fig,
                                           VisualizeGait.updateGait,
                                           len(gait_data),
                                           fargs=(entryFrames, ax, lines, xmin, ymin, zmin, xmax, ymax, zmax),
                                           interval=50, blit=False)

        if self.returnVideo:
            # https://github.com/pytorch/pytorch/issues/33226
            return line_ani.to_html5_video()
        else:
            plt.show()
