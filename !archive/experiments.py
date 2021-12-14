import json
import os
import pickle

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# from cvmdl import *
from . import * 
from . import hooks
from . import learner_callbacks
from . import learner
from . import log
from . import nets
from . import plotters
from . import samplers
from . import utils
# import hooks
# import learner
# import learner_callbacks
# import log
# import nets
# import plotters
# import samplers
# import utils

# import reports


default_lr = 0.0005
default_momentum = 0.9
default_batch_size = 64
default_train_batch_size = default_batch_size
default_test_batch_size = 1000
default_nr_epochs = 10

force_overwrite_memoziation = True  # Set this to true to recompute everything and overwrite all existing memoizations.


# WANDB Initialization
# wandb.init(project="test-project", name="test-name")
# wandb.config.dropout = 0.2
# wandb.config.hidden_layer_size = 128
# WANDB Initialization END


# n_epochs = 3
# batch_size_train = 64
# batch_size_test = 1000
# learning_rate = 0.0005
# momentum = 0.9
# log_interval = 10


class ExperimentManager:

    def __init__(self, experiment_id, suggested_architecture, suggested_optimizer, Task,
                 show_plots=False):
        assert issubclass(suggested_architecture, nets.CvMNet)
        assert issubclass(suggested_optimizer, torch.optim.Optimizer)
        self.suggested_architecture, self.suggested_optimizer, = suggested_architecture, suggested_optimizer
        self.Task = Task
        task = Task()
        self.loss_fun = task.loss_fun
        self.train_data, self.test_data = task.data_set
        # self.logger = log.Logger()
        self.task_input_shape = task.input_shape
        self.task_output_shape = task.output_shape
        # self.experiment_path
        self.experiment_id = experiment_id
        self._show_plots = show_plots

        # set initial computation state, then overwrite it if a json file already existed, otherwise create the json
        # file and store the initial state.
        self.computation_state = {"kaas": 5}
        self.try_create_comp_state_file()

    def run_all_experiments(self):
        self.show_image_samples()
        self.run_basic_tests()
        self.run_main_training()
        if self._show_plots:
            self.open_results_dir()
        else:
            pass  # not sure how to show all of them.
        # utils.play_debug_sound("experiment run has finished")

    def test_messing_around_with_hooks(self):  # just to test if hooks class works and how to use it .
        # learner = self.new_basic_learner(name="Main net",
        # net=nets.TestNet(self.task_input_shape, self.task_output_shape))
        class TestNet(nets.CvMNet):
            def __init__(self):
                super().__init__([2], [2])
                self.linnet1 = nets.LinClassifier([2], [4])
                self.linnet2 = nets.LinClassifier([4], [2])

            def _forward(self, x):
                x = self.linnet1(x)
                x = self.linnet2(x)
                return x

        net = TestNet()
        learner = self.new_basic_learner(name="Main net", net=net)
        test_hook = hooks.PrintEverythingHooks(learner.net)
        print("hooks:", net._backward_hooks)
        # x, y = next(iter(learner.train_loader))
        x, y = torch.Tensor([[1., 2.]]), torch.LongTensor([0])
        print("y:", y)
        out = learner.net(x)
        loss = self.loss_fun()(out, y)
        loss.backward()
        test_hook.remove()
        print("kaas")
        learner.net(x)

    def run_main_training(self):
        print("STARTING: run_main_training")

        # def _do():
        #     _learner = self.new_basic_learner(name="Main net")  # _learner.fit(default_nr_epochs)
        #     _learner.fit(3)
        #     return _learner
        #
        # main_learner = self._with_memoization(_do, overwrite=False)
        # self.plot_results(main_learner)

        _learner = self.new_basic_learner(name="Main net")
        self.run_experiments_on_learner(_learner)

    def run_basic_tests(self):
        self.overfit_single_batches()
        self.fit_linear_classifier()

        # baseline benchmark with linear regression or averaging and then regression.

    

    def overfit_single_batches(self):
        def _do():
            _losses_per_run, _counter_per_run = [], []
            lp_data_list = []
            for i in range(0, 8):
                batch_size = 2 ** i
                lpd = self.average_overfit_single_batch(batch_size, 1)
                # print(f"Losses for batch_size = {batch_size} computed")
                lp_data_list.append(lpd)
                # _losses_per_run.append(losses)
                # _counter_per_run.append(counter)
            return lp_data_list

        print("STARTING: overfit_single_batches")
        lp_data_list = self._with_memoization(_do)
        print(lp_data_list)
        # breakpoint()
        plotters.plot_loss_and_log_multiple(lp_data_list, dir_path=self.get_results_dir(),
                                            file_name="overfit_single_batches")

    def fit_linear_classifier(self):
        print("STARTING: fit_linear_classifier")

        net = nets.LinClassifier(self.task_input_shape, self.task_output_shape)
        _learner = self.new_basic_learner(name="Linear Classifier", net=net)
        self.run_experiments_on_learner(_learner)

    def run_experiments_on_learner(self, my_learner, plot_results=True):
        def _do():
            my_learner.fit(default_nr_epochs)
            return my_learner

        my_learner = self._with_memoization(_do, my_learner)
        if plot_results:
            self.plot_results(my_learner)

    def run_experiments_on_learners(self, my_learners, experiment_name, plot_individual_results=True):
        for lr in my_learners:
            self.run_experiments_on_learner(lr, plot_results=plot_individual_results)

        # breakpoint()
        self.plot_comparative_results(experiment_name, my_learners)

    def overfit_single_batch(self, batch_size):
        learner = self.new_basic_uniform_learner(name=f"Overfit {batch_size}", batch_size=batch_size)
        learner.fit(n_epochs=1)
        lpd = learner.SLLoggerCallback.get_loss_plot_data(count_per_batch=True)
        return lpd

    def average_overfit_single_batch(self, batch_size, nr_samples):
        lossesl = None
        for i in range(0, nr_samples):
            lpd = self.overfit_single_batch(batch_size)
            losses = lpd.train_losses

            if lossesl is None:
                lossesl = losses
            else:
                lossesl = [lossesl[i] + losses[i] for i in range(0, len(losses))]
            # counterl += counter
        losses = [i / nr_samples for i in lossesl]
        lpd = log.LossPlotData("", lpd.train_counter, losses)
        # counter = [i / nr_samples for i in counterl]
        return lpd

    def new_basic_uniform_learner(self, name, batch_size=64):
        task = self.Task()
        loader = samplers.HackySingleBatchDataLoader(self.train_data, batch_size, nr_batches=100)
        task.set_data_loaders((loader, loader))
        net = self.suggested_architecture(self.task_input_shape, self.task_output_shape)
        return learner.SupervisedLearner(name=name,
                                         net=net,
                                         optimizer=optim.SGD(net.parameters(), default_lr, default_momentum),
                                         task=task,
                                         callbacks=[learner_callbacks.SLLoggerCallback,
                                                    learner_callbacks.GradientLoggerCallback,
                                                    learner_callbacks.WeightInitializerCallback])

    def new_basic_learner(self, name, net=None, batch_size_train=default_train_batch_size,
                          batch_size_test=default_test_batch_size):
        if net is None:
            net = self.suggested_architecture(self.task_input_shape, self.task_output_shape)
        print("new basic learner")
        return learner.SupervisedLearner(name=name,
                                         net=net,
                                         optimizer=optim.SGD(net.parameters(), default_lr, default_momentum),
                                         task=self.Task(),
                                         callbacks=[learner_callbacks.SLLoggerCallback,
                                                    learner_callbacks.GradientLoggerCallback,
                                                    learner_callbacks.WeightInitializerCallback])

    def plot_results(self, lrn):
        # breakpoint()
        lrn.SLLoggerCallback.plot_loss(dir_path=self.get_results_dir(), file_name=f"{lrn.name}_loss")
        # self.plot_loss(file_name=f"{lrn.name}_loss", lrn=lrn)
        self.plot_gradients(file_name=f"{lrn.name}_gradients", lrn=lrn)
        # self.plot_activation_histogram(file_name=f"{lrn.name}_act_histograms", lrn=lrn)

    def plot_comparative_results(self, plot_name, my_learners):
        self.plot_comparative_loss(file_name="comparative_loss", my_learners=my_learners, plot_name=plot_name)

    def plot_comparative_loss(self, file_name, plot_name, my_learners):
        lp_data_list = [lrn.SLLoggerCallback.get_loss_plot_data() for lrn in my_learners]
        fig = plt.figure()

        for lp_data in lp_data_list:
            train_losses_ma = utils.moving_average(lp_data.train_losses, 100)
            plt.plot(lp_data.train_counter, train_losses_ma, zorder=1)

        plt.legend([lrn.name for lrn in my_learners], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('Cross-entropy loss')

        self.plt_savefig(file_name, plot_name)

    # def plot_loss(self, file_name, lrn):
    #     lp_data = lrn.LoggerCallback.get_loss_plot_data()
    #     fig = plt.figure()
    #
    #     train_losses_ma = utils.moving_average(lp_data.train_losses, 100)
    #
    #     plt.plot(lp_data.train_counter, lp_data.train_losses, alpha=0.3, color='blue', zorder=0)
    #     plt.plot(lp_data.train_counter, train_losses_ma, color='blue', zorder=1)
    #     plt.scatter(lp_data.test_counter, lp_data.test_losses, color='red', zorder=2)
    #     plt.legend(['Train Loss MA', 'Train Loss', 'Test Loss'], loc='upper right')
    #     plt.xlabel('number of training examples seen')
    #     plt.ylabel('Cross-entropy loss')
    #
    #     self.plt_savefig(file_name, lrn.name)

    def plot_gradients(self, file_name, lrn):
        lrn.GradientLoggerCallback.plot_grads()

        self.plt_savefig(file_name, lrn.name)

    def plot_activation_histogram(self, file_name, lrn):
        lrn.GradientLoggerCallback.plot_hists()
        self.plt_savefig(file_name, lrn.name)

    def show_image_samples(self):
        task = self.Task()
        train_loader, eval_data = task.data_loaders
        train_iter = iter(train_loader)
        images, labels = train_iter.next()
        # figure = plt.figure()
        num_of_images = 60
        # plt.show()
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        self.plt_savefig(f"{self.experiment_id}_image_samples")

    # =====================================================================================
    # U T I L I T I E S
    # =====================================================================================

    def plt_savefig(self, file_name, sub_folder=None):
        utils.plt_savefig(file_name, self.get_results_dir(), sub_folder)

    def get_base_dir(self):
        return f"files/experiments/{self.experiment_id}"

    def get_results_dir(self):
        return f"{self.get_base_dir()}/result_graphs"

    def _with_memoization(self, method, parameters=None, overwrite=False):
        path = self.get_memoization_path(method, parameters)
        # print("path:", method, parameters)
        if (not overwrite) and (not force_overwrite_memoziation):
            print("trying to load pickle for ", method)
            try:
                with open(path, "rb") as read_file:
                    result = pickle.load(read_file)
                    print("loaded pickled file")
                    return result
            except FileNotFoundError:
                pass
        result = method()
        print("pickling file")
        with open(path, "wb") as write_file:
            pickle.dump(result, write_file)
        return result
    
    def get_memoization_path(self, method, parameters=None):
        dir_path = f"{self.get_base_dir()}/memoization"
        utils.make_dir(dir_path)
        return f"{dir_path}/{utils.hash_method_with_parameters(method, parameters)}.pickle"

    def get_comp_state_path(self):
        return f"{self.get_base_dir()}/computation_state.json"

    def save_comp_state(self):
        with open(self.get_comp_state_path(), "w") as write_file:
            json.dump(self.computation_state, write_file)

    def load_comp_state(self):
        with open(self.get_comp_state_path(), "r") as read_file:
            self.computation_state = json.load(read_file)

    def try_create_comp_state_file(self):
        if os.path.isfile(self.get_comp_state_path()):
            self.load_comp_state()
        else:
            self.save_comp_state()

    def open_results_dir(self):
        os.system(f"xdg-open {self.get_results_dir()}")
