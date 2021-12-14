import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init

from . import learner
from . import log
from . import plotters
from . import utils
from .learner import LearnerCallback
from .learner import LearnerCallbackWithHooks as LearnerCallbackWithHooks

# from learner import LearnerCallbackWithHooks as Learner

callback_events = ["after_init",
                   "before_fit", "after_fit",
                   "before_initial_test", "after_initial_test",
                   "before_epoch", "after_epoch",
                   "before_train", "after_train", "before_test", "after_test",
                   "before_batch", "after_batch", "before_episode", "after_episode",  # for ReinforcementLearners
                   "after_inference", "after_loss", "before_backward", "after_backward", "after_opt.step",
                   "after_zero_grad",
                   "flush_caches"]
assert callback_events == learner.callback_events


# TODO: I'll probably want to generalize accuracy to arbitrary metrics, and have it log an arbitrary set of metrics. In fact, it doesn't check whether accuracy is even a valid metric on the data set.

# TODO: I need to end up with a single loss and counter list for all epochs
# class TrainManagerCallback(LearnerCallback):
#     def fit_scheme(self, nr_epochs):
#         self.nr_epochs = nr_epochs
#         print("id: ", id(self))
#         print("cepoch, ", self.c_epoch)
#         self.c_epoch = self.lrnr.c_epoch
#         self.lrnr.fit(nr_epochs)
#
#     def before_fit(self):
#         print("id: ", id(self))
#         self.lrnr.c_epoch = self.c_epoch


class LoggerCallback(LearnerCallback):
    def __init__(self, lrnr=None):
        super().__init__(lrnr)
        self.train_batch_counter, self.train_sample_counter = [], []
        self.test_batch_counter, self.test_sample_counter = [], []  # both for batch losses but counted differently
        self.epoch_test_batch_counter, self.epoch_test_sample_counter = [], []  # same, but for average losses per epoch
        self.c_epoch_test_counter, self.c_epoch_test_count = [], 0
        # self.losses_for_epoch, self.counters_for_epoch = [], []
        # self.duration_for_epoch = []
        self.c_batch_count = self.c_sample_count = 0
        self.end_time = self.start_time = 0

    def before_initial_test(self):
        self.before_epoch()

    def after_initial_test(self):
        self.end_time = time.time()
        self.print_test_epoch()

    def before_epoch(self):
        self.c_epoch_test_counter, self.c_epoch_test_count = [], 0
        self.start_time = time.time()

    def after_batch(self):
        if self.lrnr.training_mode:
            self.c_sample_count += self.lrnr.c_batch_size
            self.c_batch_count += 1

            self.train_sample_counter.append(self.c_sample_count)
            self.train_batch_counter.append(self.c_batch_count)
        else:
            # breakpoint()
            # print("test")
            self.c_epoch_test_count += self.lrnr.c_batch_size

            self.c_epoch_test_counter.append(self.c_epoch_test_count)

            self.test_sample_counter.append(self.c_sample_count)
            self.test_batch_counter.append(self.c_batch_count)

    def after_test(self):
        self.epoch_test_batch_counter.append(self.c_batch_count)
        self.epoch_test_sample_counter.append(self.c_sample_count)

    def after_epoch(self):
        self.end_time = time.time()
        self.print_test_epoch()

    def after_fit(self):
        pass

    def print_test_epoch(self):
        raise NotImplementedError

    def get_loss_plot_data(self, count_per_batch=False) -> log.LossPlotData:
        raise NotImplementedError

    #     def get_activation_data(self):

    def get_learner_data(self) -> log.LearnerData:
        raise NotImplementedError

    def plot_loss(self, dir_path=None, file_name=None, show=False):
        plotters.plot_loss(self.get_loss_plot_data(), dir_path, file_name, show)


class SLLoggerCallback(LoggerCallback):

    def __init__(self, lrnr=None):
        super().__init__(lrnr)
        self.train_losses = []
        self.test_losses = []
        self.epoch_test_losses = []
        self.c_epoch_test_losses = []

        self.c_epoch_total_test_correct = 0
        self.c_epoch_accuracy = 0
        self.accuracies = []

    def before_epoch(self):
        self.c_epoch_test_losses = []
        self.setup_epoch_accuracy()
        super().before_epoch()

    def setup_epoch_accuracy(self):
        self.c_epoch_total_test_correct = 0

    def after_batch(self):
        super().after_batch()
        if self.lrnr.training_mode:
            self.train_losses.append(self.lrnr.c_loss)
        else:
            self.test_losses.append(self.lrnr.c_loss)
            self.c_epoch_test_losses.append(self.lrnr.c_loss)
            self.log_batch_accuracy()

    def after_test(self):
        super().after_test()
        self.epoch_test_losses.append(np.mean(self.c_epoch_test_losses))

        correct_preds = self.c_epoch_total_test_correct
        total_preds = self.c_epoch_test_count
        accuracy = 100. * correct_preds / total_preds
        self.c_epoch_accuracy = accuracy
        self.accuracies.append(accuracy)

    def log_batch_accuracy(self):
        pred = self.lrnr.c_out.max(1, keepdim=True)[1]  # This is the same as c_out.data.max(1, keepdim=True)[1]
        self.c_epoch_total_test_correct += torch.sum(torch.eq(pred, self.lrnr.c_batch_y.view_as(pred)))

        # accuracy

        # self.c_epoch_accuracy = self.c_epoch_total_test_correct / self.c_epoch_test_count
        # self.accuracies.append(self.c_epoch_accuracy)

    def get_loss_plot_data(self, count_per_batch=False, average_test_loss_by_epoch=True) -> log.LossPlotData:
        # breakpoint()
        if count_per_batch and average_test_loss_by_epoch:
            train_counter, test_counter = self.train_batch_counter, self.epoch_test_batch_counter
            test_losses = self.epoch_test_losses
        elif not count_per_batch and average_test_loss_by_epoch:
            train_counter, test_counter = self.train_sample_counter, self.epoch_test_sample_counter
            test_losses = self.epoch_test_losses
        elif count_per_batch and not average_test_loss_by_epoch:
            train_counter, test_counter = self.train_batch_counter, self.test_batch_counter
            test_losses = self.test_losses
        else:  # not count_per_batch and not average_test_loss_by_epoch:
            train_counter, test_counter = self.train_sample_counter, self.test_sample_counter
            test_losses = self.test_losses

        return log.LossPlotData(self.lrnr.name, train_counter, self.train_losses, test_counter, test_losses)

    def get_accuracy_data(self):
        return self.accuracies

    def get_learner_data(self) -> log.LearnerData:
        pass

    def print_test_epoch(self):
        dur = self.end_time - self.start_time
        total_test_loss = np.mean(self.c_epoch_test_losses)
        # correct_preds, total_preds = None, None
        # if isinstance(self.lrnr, learner.SupervisedLearner):
        #     correct_preds = self.c_epoch_total_test_correct
        #     total_preds = self.c_epoch_test_count

        correct_preds = self.c_epoch_total_test_correct
        total_preds = self.c_epoch_test_count
        log_name = f'{self.lrnr.name}: '
        epoch = f'Epoch {self.lrnr.c_epoch} (B={self.c_batch_count}, S={self.c_epoch_test_count}): '
        # samples = f'Samples: {total_test_samples}'
        loss = f'Test loss: {total_test_loss:.4f}, '
        # loss = f'Test avg loss:{test_loss * total_test_samples:.0f}/{total_test_batches}({test_loss:.4f}), '
        # accuracy = ''
        # if correct_preds is not None:
        #     accuracy = f'Accuracy: {100. * correct_preds / total_preds:.2f}%, '
        accuracy = f'Accuracy: {100. * correct_preds / total_preds:.2f}%, '
        duration = f'Dur: {dur:.1f} s'
        print(log_name, epoch + loss + accuracy, duration)
        #
        # print_test_epoch(log_name=self.lrnr.name,
        #                  n_batches=self.c_batch_count,
        #                  epoch=self.lrnr.c_epoch,
        #                  test_loss=total_test_loss,
        #                  total_test_samples=self.c_epoch_test_count,
        #                  correct_preds=correct_preds,
        #                  # total_preds=total_preds,
        #                  duration_ns=dur)


def print_test_epoch(log_name, n_batches, epoch, test_loss, total_test_samples, correct_preds, duration_ns):
    log_name = f'{log_name}: '
    epoch = f'Epoch {epoch} (#B={n_batches},#S={total_test_samples}): '
    # samples = f'Samples: {total_test_samples}'
    loss = f'Test loss: {test_loss:.4f}, '
    # loss = f'Test avg loss:{test_loss * total_test_samples:.0f}/{total_test_batches}({test_loss:.4f}), '
    accuracy = ''
    if correct_preds is not None:
        accuracy = f'Accuracy: {100. * correct_preds / total_test_samples:.2f}%, '
    duration = f'Dur: {duration_ns:.1f} s'
    print(log_name, epoch + loss + accuracy, duration)


class RLLoggerCallback(LoggerCallback):
    def __init__(self, lrnr=None):
        super().__init__(lrnr)
        self.train_values, self.test_values = [], []
        self.c_test_values = []

    def after_episode(self):
        super().after_batch()
        if self.lrnr.training_mode:
            self.train_values.append(self.lrnr.c_value)
        else:
            self.test_values.append(self.lrnr.c_value)
            self.c_test_values.append(self.lrnr.c_value)

    # def after_test(self):
    #     if not self.lrnr.training_mode and len(self.c_test_values) > 0:
    #         self.test_values.append(np.mean(self.c_test_values))
    #         self.c_test_values = []

    def print_test_epoch(self):
        dur = self.end_time - self.start_time
        # total_test_loss = sum(self.c_epoch_test_losses)
        average_value = np.mean(self.test_values)
        print_test_epoch_rl(log_name=self.lrnr.name,
                            n_batches=self.c_batch_count,
                            epoch=self.lrnr.c_epoch,
                            # test_loss=total_test_loss,
                            total_test_episodes=self.c_epoch_test_count,
                            average_value=average_value,
                            duration_ns=dur)

    def get_loss_plot_data(self, count_per_batch=False) -> log.LossPlotData:
        # breakpoint()
        if count_per_batch:
            return log.LossPlotData(self.lrnr.name, self.train_batch_counter, self.train_values,
                                    self.test_batch_counter, self.test_values)
        else:
            return log.LossPlotData(self.lrnr.name, self.train_sample_counter, self.train_values,
                                    self.test_sample_counter, self.test_values)


def print_test_epoch_rl(log_name, n_batches, epoch, total_test_episodes, average_value,
                        duration_ns):
    log_name = f'{log_name}: '
    epoch = f'Epoch {epoch} ({n_batches} batches): '
    # loss = f'Test avg loss: {test_loss * total_test_episodes:.4f}/{total_test_episodes} ({test_loss:.4f}), '
    value = f'Test avg value:{average_value * total_test_episodes:.0f}/{total_test_episodes}({average_value:.4f}), '
    duration = f'Dur: {duration_ns:.1f} s'
    print(log_name, epoch + value, duration)


class HooksCallback:
    pass


# ===============================================================================================
# ===
# ===

class GradAndActLoggerCallback(LearnerCallbackWithHooks):
    pass


class GradAndActProfilerCallback(GradAndActLoggerCallback):
    def __init__(self, lrn):
        super().__init__(lrn)
        self.modules = self.get_modules()
        self.module_nr = {}
        for i, (n, m) in enumerate(self.modules):
            self.module_nr[str(id(m))] = i
        # print("modules: ", self.modules)
        # print("module_nr")
        self._layer_param_grad_mean, self._layer_param_grad_m2 = [], []
        self._layer_out_grad_mean, self._layer_out_grad_m2 = [], []
        self._layer_out_act_mean, self._layer_out_act_m2 = [], []

    def get_modules(self):
        return utils.named_flatten_module(self.lrnr.net)

    def reset_learner(self, lrnr):
        super().reset_learner(lrnr)
        self.modules = self.get_modules()

    def before_test(self):
        # print("before_test from callback")
        ep = self.lrnr.c_epoch
        self._layer_param_grad_mean.insert(ep, [[] for _ in self.modules])
        self._layer_param_grad_m2.insert(ep, [[] for _ in self.modules])
        self._layer_out_grad_mean.insert(ep, [0 for _ in self.modules])
        self._layer_out_grad_m2.insert(ep, [0 for _ in self.modules])
        self._layer_out_act_mean.insert(ep, [0 for _ in self.modules])
        self._layer_out_act_m2.insert(ep, [0 for _ in self.modules])
        return True

    # def after_test(self):
    # n = self.lrnr.c_batch_nr

    @torch.utils.hooks.unserializable_hook
    def hook_on_output_tensor(self, m, grad):
        if self.lrnr.training_mode: return
        ep = self.lrnr.c_epoch
        i = self.module_nr[str(id(m))]
        n = self.lrnr.c_batch_nr + 1
        data = grad.data.numpy()
        self._layer_out_grad_mean[ep][i] += (data - self._layer_out_grad_mean[ep][i]) / n
        self._layer_out_grad_m2[ep][i] += (np.multiply(data, data) - self._layer_out_grad_m2[ep][i]) / n
        # print("hook on output tensor from callback")

    def forward_hook(self, m, inp, outp: torch.Tensor):
        if self.lrnr.training_mode: return
        ep = self.lrnr.c_epoch
        i = self.module_nr[str(id(m))]
        n = self.lrnr.c_batch_nr + 1
        m = outp.data.numpy()
        m2 = np.multiply(m, m)
        # breakpoint()
        m = np.mean(m, axis=0)
        m2 = np.mean(m2, axis=0)
        self._layer_out_act_mean[ep][i] += (m - self._layer_out_act_mean[ep][i]) / n
        self._layer_out_act_m2[ep][i] += (m2 - self._layer_out_act_m2[ep][i]) / n
        # breakpoint()
        # print("forward hook from callback")

    def after_backward(self):
        if self.lrnr.training_mode: return
        ep = self.lrnr.c_epoch
        n = float(self.lrnr.c_batch_nr + 1)
        # print("n: ", n)
        for i, (_, m) in enumerate(self.modules):
            # grads = []
            if not self._layer_param_grad_mean[ep][i]:
                # self._layer_param_grad_mean[ep][i], self._layer_param_grad_m2[ep][i] = [], []
                for _, p in enumerate(m.parameters()):
                    try:
                        grad = p.grad.numpy()
                        self._layer_param_grad_mean[ep][i].append(grad)
                        self._layer_param_grad_m2[ep][i].append(np.multiply(grad, grad))
                    except:
                        breakpoint()
                continue
            for j, p in enumerate(m.parameters()):
                grad = p.grad.numpy()
                self._layer_param_grad_mean[ep][i][j] += (grad - self._layer_param_grad_mean[ep][i][j]) / n
                self._layer_param_grad_m2[ep][i][j] += (np.multiply(grad, grad) - self._layer_param_grad_m2[ep][i][
                    j]) / n

            # if not grads: continue

            # for j, p in enumerate(m.parameters()):
            #     self._layer_param_grad_mean[ep][i][j] += (p.grad.numpy() - self._layer_param_grad_mean[ep][i][j]) / n
            #     self._layer_param_grad_m2[ep][i][j] += (np.multiply(p.grad.numpy(), p.grad.numpy()) -
            #                                             self._layer_param_grad_m2[ep][i][j]) / n

            # if self._layer_param_grad_mean[i] is None:
            #     self._layer_param_grad_mean[i], self._layer_param_grad_m2[i] = [], []
            #     for _, p in enumerate(m.parameters()):
            #         self._layer_param_grad_mean[i].append(p.grad.numpy())
            #         self._layer_param_grad_m2[i].append(np.multiply(p.grad.numpy(), p.grad.numpy()))
            # else:
            #     for j, p in enumerate(m.parameters()):
            #         self._layer_param_grad_mean[i][j] += p.grad.numpy()
            #         self._layer_param_grad_m2[i][j] += np.multiply(p.grad.numpy(), p.grad.numpy())

    def after_inference(self):
        return True

    def get_data(self, tidy=False):
        param_grad = (self._layer_param_grad_mean, self._layer_param_grad_m2)
        node_grad = (self._layer_out_grad_mean, self._layer_out_grad_m2)
        node_act = (self._layer_out_act_mean, self._layer_out_act_m2)
        # print("bizar: ", param_grad, node_grad, node_act)
        if not tidy: return param_grad, node_grad, node_act


def grad_and_act_profile(lrnr: learner.Learner):
    pass
    # put the ProfilerCallback on it and output the data.


class GradientLoggerCallback(LearnerCallbackWithHooks):

    def __init__(self, lrn):
        super().__init__(lrn)
        self.modules = utils.named_flatten_module(self.lrnr.net)

        self.layers_param_grad_means = [[] for _ in self.modules]
        self.layers_param_grad_sds = [[] for _ in self.modules]
        self.layers_out_grad_means = [[] for _ in self.modules]
        self.layers_out_grad_sds = [[] for _ in self.modules]
        self.layers_out_act_means = [[] for _ in self.modules]
        self.layers_out_act_sds = [[] for _ in self.modules]
        self.layers_out_act_hist = [[] for _ in self.modules]

        self._layer_out_grad_mean = {}  # {id(m): None for m in self.modules}
        self._layer_out_grad_sd = {}
        self._layer_out_act_sd = {}
        self._layer_out_act_mean = {}
        self._layer_out_act_hist = {}
        # self.gradient_hook = self.GradientLoggerHook(self)

    # HOOKS
    def hook_on_output_tensor(self, m, grad):
        self._layer_out_grad_mean[str(id(m))] = grad.data.mean().item()
        self._layer_out_grad_sd[str(id(m))] = grad.data.std().item()
        # breakpoint()

    def forward_hook(self, m, inp, outp: torch.Tensor):
        self._layer_out_act_mean[str(id(m))] = outp.data.mean().item()
        self._layer_out_act_sd[str(id(m))] = outp.data.std().item()
        # outp.data.cpu().
        self._layer_out_act_hist[str(id(m))] = outp.data.cpu().histc(40, -10, 10)
        # breakpoint()

    # LEARNER CALLBACKS

    # def before_fit(self):
    #     pass

    def after_backward(self):
        assert self.lrnr.net.training
        for i, (n, m) in enumerate(self.modules):
            params = m.parameters()
            if params:
                _param_values = []
                for p in params:
                    _param_values.extend(list(p.grad.numpy().flatten()))
                self.layers_param_grad_means[i].append(np.mean(_param_values))
                self.layers_param_grad_sds[i].append(np.std(_param_values))
            self.layers_out_grad_means[i].append(self._layer_out_grad_mean[str(id(m))])
            self.layers_out_grad_sds[i].append(self._layer_out_grad_sd[str(id(m))])

            self.layers_out_act_means[i].append(self._layer_out_act_mean[str(id(m))])
            self.layers_out_act_sds[i].append(self._layer_out_act_sd[str(id(m))])
            self.layers_out_act_hist[i].append(self._layer_out_act_hist[str(id(m))])

    def after_fit(self):
        # self.plot_grads("grads")
        pass

    def plot_grads(self):
        # fig = plt.figure(figsize=(80, 6))
        fig, ax = plt.subplots(3, 2)
        print(ax)
        # (axpgm, axogm, axom), (axpgsd, axogsd, axosd) = ax
        (axpgm, axpgsd), (axogm, axogsd), (axom, axosd) = ax
        for i, (name, m) in enumerate(self.modules):
            axpgm.plot(self.layers_param_grad_means[i])
            axpgsd.plot(self.layers_param_grad_sds[i])
            axogm.plot(self.layers_out_grad_means[i])
            axogsd.plot(self.layers_out_grad_sds[i])
            axom.plot(self.layers_out_act_means[i])
            axosd.plot(self.layers_out_act_sds[i])
        for i in range(0, 3):
            for a in ax[i]:
                a.legend([name for (name, m) in self.modules], loc="upper right")

        plt.xlabel('number of backward passes')

        axpgm.set_ylabel('layer mean\n parameter grad', rotation=0, horizontalalignment="right")
        axpgsd.set_ylabel('layer sd\n parameter grad', rotation=0, horizontalalignment="right")
        axogm.set_ylabel('layer mean\n activation grad', rotation=0, horizontalalignment="right")
        axogsd.set_ylabel('layer sd\n activation grad', rotation=0, horizontalalignment="right")
        axom.set_ylabel('layer mean\n activation', rotation=0, horizontalalignment="right")
        axosd.set_ylabel('layer sd\n activation', rotation=0, horizontalalignment="right")
        # plt.tight_layout()
        fig.set_size_inches(20, 10)
        # plt.show()
        # show_plots()

    def plot_hists(self):
        hists = ...

        def get_hist(histogram):
            return torch.stack(histogram).t().float().log1p()

        fig, axes = plt.subplots(len(self.modules), 1, figsize=(15, 6))
        try:
            iter(axes)
        except TypeError:
            axes = [axes]
        for ax, (i, (name, m)) in zip(axes, enumerate(self.modules)):
            ax.imshow(get_hist(self.layers_out_act_hist[i]), origin='lower')
            ax.axis('off')
        plt.tight_layout()

        # breakpoint()
    # def _create_hooks(self):
    # return GradientLoggerHook(self)


class WeightInitializerCallback(LearnerCallback):
    def after_init(self):
        for l in self.lrnr.net.modules():
            try:
                init.kaiming_normal_(l.conv1.weight)
            except AttributeError:
                pass
            try:
                l.bias.data.zero_()
            except AttributeError:
                pass
            # breakpoint()
