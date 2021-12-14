import json
import os
import pickle
import random
from typing import List
from dataclasses import dataclass
from datetime import datetime

# import os
cwd = os.getcwd()
print ("cwd pleh: ", cwd)

import torch
import torch.utils.data
from IPython.display import Video
from torch import nn

from . import utils
from .hooks import Hooks

class EventException(Exception):
    pass


callback_events = ["after_init",
                   "before_fit", "after_fit",
                   "before_initial_test", "after_initial_test",
                   "before_epoch", "after_epoch",
                   "before_train", "after_train", "before_test", "after_test",
                   "before_batch", "after_batch", "before_episode", "after_episode",
                   "after_inference", "after_loss", "before_backward", "after_backward", "after_opt.step",
                   "after_zero_grad",
                   "flush_caches"]


class LearnerCallback:
    def __init__(self, lrnr):
        self.lrnr: Learner = lrnr

    def __call__(self, event_string):
        return self._do_callback(event_string)

    def _do_callback(self, event_string):
        fun = getattr(self, event_string, None)
        if fun is not None: return fun()

    def __repr__(self):
        return self.__class__.__name__

    def reset_learner(self, lrnr):
        self.lrnr = lrnr

    def added_to_learner(self, lrnr):
        self.reset_learner(lrnr)


class LearnerCallbackWithHooks(LearnerCallback, Hooks):
    def __init__(self, lrnr):
        LearnerCallback.__init__(self, lrnr)
        Hooks.__init__(self, lrnr.net)

    #
    # def _create_hooks(self):
    #     """implement this in a subclass: Create a Hooks class and return it."""
    #     raise NotImplementedError

@dataclass
class LearnerSpec:
    def __init__(self, path):
        self.path= path
        
    def __call__(self, tryload=True,directory=None, file_name =None):
        if directory is None: directory = self.path
        if tryload:
            try:
                lrnr =self.load(directory, file_name)
                return lrnr
            except FileNotFoundError:
                pass
        return self.make()

    def name(self):
        pass

    def make(self):
        pass

    def load(self, directory, file_name=None):
        if directory is None: directory = self.path
        path = _get_path_from_dir_file(self.name(), directory, file_name, "_learner.pkl")
        with open(path, "rb") as read_file:
            loaded_learner = pickle.load(read_file)
            time = _str_time()
            print(f"{time} Successfully loaded learner: {loaded_learner.name}")
            return loaded_learner

    def delete_saved_files(self, directory=None, file_name=None):
        if directory is None: directory = self.path
        
        paths = [_get_path_from_dir_file(self.name(), directory, file_name, x) for x in
                 ("_learner.pkl", "_params.pt", "_metadata.json")]
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _get_path_from_dir_file(name, directory, file_name=None, post_script=""):
    if file_name is None:
        file_name = f"{name}{post_script}"
    return directory + file_name

def _str_time():
    return datetime.now().strftime("%b-%d %H:%M:%S")

class Learner:
    """@DynamicAttrs"""
    c_n_epochs: int
    c_epoch: int
    c_batch_nr: int
    c_batch_x: torch.Tensor
    c_batch_y: torch.Tensor
    c_dl: torch.utils.data.DataLoader
    training_mode: bool
    callbacks: List[LearnerCallback]
    net: nn.Module

    def __init__(self, name, net, optimizer, callbacks=None):
        super().__init__()
        # print("init learner")
        self.name = name
        self.net, self.optimizer = net, optimizer

        self.architecture = net.__class__
        self.callbacks = []
        assert issubclass(self.architecture, nn.Module)
        assert isinstance(net, nn.Module)
        if callbacks is None: callbacks = []
        self.add_callbacks(callbacks)
        self.training_mode = False
        self.c_epoch = 0
        self.c_n_epochs = 0
        self.c_loss_tensor, self.c_loss = None, None

        self("after_init")

    def __repr__(self):
        return f"NAME: {self.name}\n NET: {self.net}\n OPTIMIZER: {self.optimizer}\n CALLBACKS: {self.callbacks}\n"

    def flush_caches(self):
        self.c_loss_tensor = None
        self("flush_caches")

    def save(self, directory, file_name=None):
        self.flush_caches()
        self.save_self(directory, file_name)
        self.save_parameters(directory, file_name)
        self.save_metadata(directory, file_name)
        time = _str_time()
        print(f"{time} Saved learner data: {self.name}")

    def save_self(self, directory, file_name=None):
        utils.make_dir(directory)
        path = _get_path_from_dir_file(self.name, directory, file_name, "_learner.pkl")
        with open(path, "wb") as write_file:
            pickle.dump(self, write_file)

    def load_self(self, directory, file_name=None):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_learner.pkl")
        try:
            with open(path, "rb") as read_file:
                loaded_learner = pickle.load(read_file)
                print(f"Successfully loaded: load_self: {self.name}")
                return loaded_learner
        except FileNotFoundError:
            print(f"File not found when trying to load pickled self: {self.name}")
            return self

    def save_parameters(self, directory, file_name=None):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_params.pt")
        utils.make_dir(directory)
        torch.save(self.net.state_dict(), path)

    def load_parameters(self, directory, file_name=None, continue_if_fails=True):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_params.pt")
        if continue_if_fails:
            try:
                self.net.load_state_dict(torch.load(path))
                print(f"Successfully loaded: load_parameters {self.name}")
            except FileNotFoundError:
                print(f"File not found when trying to load parameters: {self.name}")
                return
        else:
            self.net.load_state_dict(torch.load(path))

    def save_callbacks(self, directory, file_name=None):
        self.save_self(directory, file_name)

    def load_callbacks(self, directory, file_name=None):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_learner.pkl")
        try:
            with open(path, "rb") as read_file:
                loaded_learner = pickle.load(read_file)
                self.remove_callbacks(self.callbacks)
                cbs = loaded_learner.callbacks
                self.add_callbacks(cbs)
                print(f"Successfully loaded: load_callbacks: {self.name}")
        except FileNotFoundError:
            print(f"File not found when trying to load callbacks: {self.name}")

    def save_metadata(self, directory, file_name=None):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_metadata.json")
        meta_data = {"c_epoch": self.c_epoch, "c_n_epochs": self.c_n_epochs}
        with open(path, "w") as write_file:
            json.dump(meta_data, write_file)

    def load_metadata(self, directory, file_name=None):
        path = _get_path_from_dir_file(self.name, directory, file_name, "_metadata.json")
        try:
            with open(path, "r") as read_file:
                meta_data = json.load(read_file)
                self.c_epoch, self.c_n_epochs = meta_data["c_epoch"], meta_data["c_n_epochs"]
                print(f"Successfully loaded: load_metadata: {self.name}")
        except FileNotFoundError:
            print(f"file not found when trying to load meta_data: {self.name}")

    def delete_saved_files(self, directory, file_name=None):
        paths = [_get_path_from_dir_file(self.name, directory, file_name, x) for x in
                 ("_learner.pkl", "_params.pt", "_metadata.json")]
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    

    # ==================================================================================================================
    # CALLBACKS

    def __call__(self, event_string: str):
        return self._apply_callbacks(event_string)

    def _apply_callbacks(self, event_string):
        assert event_string in callback_events, event_string
        x = None
        for cb in self.callbacks:
            x = cb(event_string)
        return x

    def _with_events(self, f, event_type, ex=EventException, final=None):
        output = None
        try:
            self(f"before_{event_type}")
            output = f()
            self(f"after_{event_type}")
        except ex:
            self(f"except_{event_type}")
        if final is not None: final()
        return output

    def add_callbacks(self, cbs):
        list(map(self.add_callback, cbs))

    def remove_callbacks(self, cbs):
        list(map(self.remove_callback, cbs))

    def add_callback(self, cb):
        # print("adding callback: ", cb)
        if isinstance(cb, type): cb = cb(self)
        assert isinstance(cb, LearnerCallback), cb
        cb.added_to_learner(self)
        self.callbacks.append(cb)
        setattr(self, cb.__class__.__name__, cb)
        # I don't know if the above setattr actually works. EDIT: I think it does.

    def remove_callback(self, cb):
        if isinstance(cb, type):
            self.remove_callbacks(self._grab_cbs(cb))
        else:
            assert isinstance(cb, LearnerCallback)
            self.callbacks.remove(cb)

    def _grab_cbs(self, cb_cls):
        return [cb for cb in self.callbacks if isinstance(cb, cb_cls)]

    def print_parameters(self):
        print(f"printing parameters of {self.name}")
        for m in self.net.parameters():
            print(m.data)

    # ==================================================================================================================
    # LOOPS

    def fit(self, n_epochs, additional=False, reset_epochs=False):
        if not additional:
            self.c_n_epochs = 0
        self.c_n_epochs += n_epochs
        if reset_epochs:
            self.c_epoch = 0
        self._with_events(self._do_fit, "fit")

    def _do_fit(self):
        if self.c_epoch == 0:
            self._with_events(self._do_test_epoch, "initial_test")
        # for epoch in range(1, self.c_n_epochs + 1):
        while self.c_epoch < self.c_n_epochs:
            self.c_epoch += 1
            self._with_events(self._do_epoch, "epoch")

    def _do_epoch(self):
        self._do_train_epoch()
        self._do_test_epoch()

    def _do_train_epoch(self):
        self.train()
        self._with_events(self._do_all_batches, "train")

    def _do_test_epoch(self):
        self.eval()
        grad = self("before_test")
        # breakpoint()
        if grad:
            self._do_all_batches()
        else:
            with torch.no_grad():
                self._do_all_batches()
        self("after_test")
        # with torch.no_grad(): self._with_events(self._do_all_batches, "test")

    def set_loss(self, loss_tensor):
        self.c_loss_tensor = loss_tensor
        self.c_loss = self.c_loss_tensor.item()
        self("after_loss")

    def backward(self):
        # breakpoint()
        self._with_events(self.c_loss_tensor.backward, "backward")
        # self("before_backward")
        # self.c_loss_tensor.backward()
        # self("after_backward")

    def optim_step(self, zero_grad=True):
        # self._with_events(self.optimizer.step, "opt.step")
        self.optimizer.step()
        self("after_opt.step")
        if zero_grad:
            self.optimizer.zero_grad()
            self("after_zero_grad")

    def train(self):
        self.net.train()
        self.training_mode = True

    def eval(self):
        # self.net.eval()
        self.net.train()
        self.training_mode = False

    def showcase_performance(self):
        raise NotImplementedError


class SupervisedLearner(Learner):

    def __init__(self, name, net, optimizer, task, callbacks=None):
        super().__init__(name, net, optimizer, callbacks)
        if isinstance(task, type):
            task = task()
        self.task = task
        self.train_loader, self.test_loader = task.data_loaders
        self.loss_fun = task.loss_fun()

    def flush_caches(self):
        super().flush_caches()
        self.c_batch_x, self.c_batch_y = None, None

    def _do_all_batches(self):
        for batch_nr, (batch_x, batch_y) in enumerate(self.c_dl):
            self.c_batch_nr, self.c_batch_x, self.c_batch_y = batch_nr, batch_x, batch_y
            self.c_batch_size = self.c_batch_x.shape[0]
            # Haven't tested the above line
            self._with_events(self._do_batch, "batch")

    def _do_batch(self):
        self.c_out = self.net(self.c_batch_x)
        _do_backward_ = self("after_inference")
        # breakpoint()
        self.set_loss(self.loss_fun(self.c_out, self.c_batch_y))

        if self.training_mode or _do_backward_:
            self.backward()
        if self.training_mode:
            self.optim_step()

    # ==================================================================================================================
    # SET MODES

    def train(self):
        super().train()
        self.c_dl: torch.utils.data.DataLoader = self.train_loader

    def eval(self):
        super().eval()
        self.c_dl = self.test_loader

    def __repr__(self):
        return f"NAME: {self.name}\n NET: {self.net}\n OPTIMIZER: {self.optimizer}\n LOSS_FUN: " \
               f"{self.loss_fun}\n CALLBACKS: {self.callbacks}\n"

    def showcase_performance(self):
        raise NotImplementedError


