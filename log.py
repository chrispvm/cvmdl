import numpy as np
import math

class LossPlotData:
    def __init__(self, name, train_counter, train_losses, test_counter=None, test_losses=None):
        self.name = name
        # self.loss_name = loss_name

        self.train_losses = train_losses
        self.train_counter = train_counter
        self.test_losses = test_losses
        self.test_counter = test_counter
        if test_counter is None:
            self.test_losses, self.test_counter = [], []

    def unpack(self):
        return self.train_losses, self.train_counter, self.test_losses, self.test_counter


def average_lpdata(losses,counter,averaging_range):
    # somehow, this is duplicating the first entry in the series
    losses_average, counter_average=[losses[0]], [counter[0]]
    losses, counter =losses[1:], counter[1:]
    n = math.floor(len(losses)/averaging_range)
    r = len(losses) % averaging_range
    for i in range(0,n):
        losses_average.append(np.mean(losses[i*averaging_range:(i+1)*averaging_range]))
        counter_average.append(counter[i*averaging_range])
    if not r==0:
        losses_average.append(np.mean(losses[n*averaging_range:]))
        counter_average.append(counter[n*averaging_range])
    return losses_average, counter_average

class MetricData:
    def __init__(self, name, counter, metric):
        self.name = name
        self.counter = counter
        self.metric = metric


class LearnerData:
    def __init__(self, name, loss_plot_data, metric_datas):
        self.name = name
        self.loss_plot_data = loss_plot_data
        self.metric_datas = metric_datas
