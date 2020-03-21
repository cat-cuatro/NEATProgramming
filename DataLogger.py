import matplotlib.pyplot as plot
import numpy as np

class data(object):
    def __init__(self):
        self.dataLabels = []
        self.dataValues = []
        self.entries = 0

    def recordResult(self, datum, label):
        adjustedLabel = (int)(label/100)
        self.dataLabels.append(adjustedLabel)
        self.dataValues.append(datum)
        self.entries += 1

    def barGraphData(self):
        highestYValue = np.arange(len(self.dataValues))
        plot.bar(highestYValue, self.dataValues, align='center', alpha=0.5)
        plot.xticks(highestYValue, self.dataLabels, rotation=90)
        plot.ylabel('Reward')
        plot.xlabel('Epochs Trained (x 100)')
        plot.title('Testing Reward vs. Epochs trained, 0.95 epsilon multiplier')
        plot.show()