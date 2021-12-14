import os

import torch
import torch.nn as nn
import torchvision

current_directory = os.getcwd()


class Task:
    pass


class SupervisedLearningTask(Task):
    def __init__(self, loss_fun, data_set, input_shape, output_shape):
        self.loss_fun = loss_fun
        self.data_set = data_set
        self.train_data, self.test_data = data_set
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_loaders = self.make_data_loaders()

    # def get_data_loaders(self):
    #     return self.data_loaders

    def set_data_loaders(self, data_loaders):
        self.data_loaders = data_loaders

    def make_data_loaders(self, batch_size_train=64, batch_size_test=64):
        return self._make_data_loaders(batch_size_train=batch_size_train,
                                       batch_size_test=batch_size_test,
                                       expected_train_data_len=None,
                                       expected_test_data_len=None)

    def _make_data_loaders(self, batch_size_train, batch_size_test, return_datasets=False,
                           expected_train_data_len=None, expected_test_data_len=None, num_workers=4,
                           drop_last_batch=True):
        train_data, test_data = self.data_set
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=num_workers, drop_last=drop_last_batch)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True, pin_memory=True,
                                                  num_workers=num_workers, drop_last=drop_last_batch)
        assert type(train_loader) is torch.utils.data.DataLoader and type(test_loader) is torch.utils.data.DataLoader
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        # print(len(train_loader.dataset), len(test_loader.dataset))
        assert (expected_train_data_len is None or len(train_loader.dataset) == expected_train_data_len) and (
                expected_test_data_len is None or len(test_loader.dataset) == expected_test_data_len)
        if return_datasets:
            return train_data, test_data, train_loader, test_loader
        else:
            return train_loader, test_loader


class MNISTTask(SupervisedLearningTask):
    def __init__(self):
        super().__init__(loss_fun=nn.CrossEntropyLoss,
                         data_set=self.load_mnist_dataset(),
                         input_shape=[1, 28, 28],
                         output_shape=[10])

    @staticmethod
    def load_mnist_dataset():
        mnist_train = torchvision.datasets.MNIST(current_directory + '/files/', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     #    torchvision.transforms.Normalize(
                                                     #        (0.1307,), (0.3081,))
                                                 ]))
        mnist_test = torchvision.datasets.MNIST(current_directory + '/files/', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    # torchvision.transforms.Normalize(
                                                    #  (0.1307,), (0.3081,))
                                                ]))
        return mnist_train, mnist_test


class RLEnv:
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class OpenAIGymEnv(RLEnv):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        self.env.optim_step(action)

    def reset(self):
        self.env.reset()


class CartPoleTask(Task):
    def __init__(self):
        super().__init__()
