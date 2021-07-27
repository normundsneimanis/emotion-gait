import numpy as np
import math


class LossHistory:
    def __init__(self, min_epochs=70, buckets=10, threshold=3, max_period=200):
        """
        Parameters
        ----------
        min_epochs : int, default: 100
            Start calculating with epoch number
        buckets : int, default: 10
            Calculate average of epochs in x buckets
        threshold: int, default: 4
            Number of buckets which should contain diverging losses
            Number of buckets which should contain falling test accuracy
        max_period: int, default: 200
            Number of last epochs used in calculation, if > this
        """
        self.min_epochs = min_epochs
        self.threshold = threshold
        self.buckets = buckets
        self.max_period = max_period
        self.testLoss = []
        self.trainLoss = []
        self.testAcc = []
        self.ended = 0

    def removeMax(self, li):
        if len(li) > self.max_period:
            li.pop(0)

    def add(self, li, value):
        if math.isnan(value):
            self.ended = 1
        else:
            li.append(value)
            self.removeMax(li)

    def addTestAcc(self, value):
        self.add(self.testAcc, value)

    def addTrainLoss(self, value):
        self.add(self.trainLoss, value)

    def addTestLoss(self, value):
        self.add(self.testLoss, value)

    def diverging(self):
        # Start checking when threshold is reached
        if len(self.trainLoss) < self.min_epochs:
            return 0
        if self.ended:
            print("LossHistory detected Nan value")
            return 1
        # Split train and test losses in 10 equal length buckets
        step = int(len(self.testLoss) / self.buckets)
        divergingOccurrances = 0
        platoOccurrances = 0
        platoDetected = 0
        accDecreasingOccurrances = 0
        prevTestLoss = 0
        testLossIncreasing = 0
        prevDiff = 0
        prevAcc = 0
        for i in range(self.buckets):
            diff = np.mean(self.testLoss[step*i:step*i+step-1]) - np.mean(self.trainLoss[step*i:step*i+step-1])
            acc = np.mean(self.testAcc[step*i:step*i+step-1])
            testLoss = np.mean(self.testLoss[step*i:step*i+step-1])
            if diff < 0:
                diff = 0
            # Detect loss plato situation
            if i >= self.buckets - self.threshold - int(self.threshold / 2):
                currentTrainLossAverage = np.mean(self.trainLoss[step*i:step*i+step-1])
                prevTrainLossAverage = np.mean(self.trainLoss[step*(i-1):step*i-1])
                if abs(currentTrainLossAverage - prevTrainLossAverage) < currentTrainLossAverage / 1000:
                    platoOccurrances += 1
            if i < self.buckets - self.threshold - 1:
                continue
            if i == self.buckets - self.threshold - 1:
                prevAcc = acc
                prevDiff = diff
                continue

            if i >= self.buckets - self.threshold:
                if prevTestLoss < testLoss:
                    testLossIncreasing += 1
            prevTestLoss = testLoss
            if i >= self.buckets - self.threshold:
                if diff > prevDiff:
                    divergingOccurrances += 1
                prevDiff = diff
                if acc < prevAcc:
                    accDecreasingOccurrances += 1
                prevAcc = acc

        if platoOccurrances >= self.threshold + int(self.threshold / 2):
            # Plato detected if last epoch average loss is the same as about middle epoch
            i = self.buckets - self.threshold - int(self.threshold / 2)
            middleLossAverage = np.mean(self.trainLoss[step*i:step*i+step-1])
            if abs(currentTrainLossAverage - middleLossAverage) < currentTrainLossAverage / 1000:
                platoDetected = 1

        if testLossIncreasing >= self.threshold or accDecreasingOccurrances >= self.threshold or platoDetected:
            print("Diverging: %d, testLoss increasing: %d, Accuracy decreasing: %d, Plato: %d (%d)" %
                  (divergingOccurrances, testLossIncreasing, accDecreasingOccurrances, platoDetected, platoOccurrances))
            return 1
        return 0
