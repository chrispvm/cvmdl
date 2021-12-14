import math

import torch
from torch.utils.data import sampler
from torch.utils.data.sampler import Sampler


class HackySingleBatchDataLoader(torch.utils.data.DataLoader):

    def __init__(self, data, batch_size, nr_batches=None, nr_samples=None):
        super().__init__(data, batch_size)
        self.data = data
        if (nr_batches is None and nr_samples is None) or (nr_batches is not None and nr_samples is not None):
            raise ValueError("you must set one and only one of nr_batches or nr_samples")
        if nr_samples is not None:
            nr_batches = int(nr_samples / batch_size)
        elif nr_batches is not None:
            nr_samples = nr_batches * batch_size
        self.nr_batches = nr_batches
        self.nr_samples = nr_samples

    def __iter__(self):
        loader = self.new_uniform_loader(batch_size=self.batch_size)
        self.batch_x, self.batch_y = next(iter(loader))
        self.cur_batch = 0
        return self

    def __next__(self):
        if self.cur_batch >= self.nr_batches:
            raise StopIteration
        self.cur_batch += 1
        return self.batch_x, self.batch_y

    def new_uniform_loader(self, batch_size):
        num_classes = len(self.data.classes)
        train_batch_sampler = HackyUniformSampler(data_set=self.data,
                                                  sampler=sampler.SequentialSampler(self.data),
                                                  num_classes=num_classes, batch_size=batch_size)
        loader = torch.utils.data.DataLoader(dataset=self.data, batch_sampler=train_batch_sampler,
                                             pin_memory=True, num_workers=0)
        return loader


class HackyUniformSampler(Sampler):
    # Use num_workers = 0 for this.
    def __init__(self, data_set, sampler, num_classes, batch_size: int) -> None:
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        self.data_set = data_set
        self.sampler = sampler
        self.num_classes = len(data_set.classes)
        self.max_per_class = int(math.ceil(batch_size / self.num_classes))
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        classes = []
        for idx in self.sampler:
            (image, label) = self.data_set[idx]
            if classes.count(label) >= self.max_per_class:
                continue
            batch.append(idx)
            classes.append(label)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # copied from BatchSampler
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore

# class MPerClassSampler(torch.data.Sampler):
#     """
#     At every iteration, this will return m samples per class. For example,
#     if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
#     each will be returned
#     """
#
#     def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
#         if isinstance(labels, torch.Tensor):
#             labels = labels.numpy()
#         self.m_per_class = int(m)
#         self.batch_size = int(batch_size) if batch_size is not None else batch_size
#         self.labels_to_indices = c_f.get_labels_to_indices(labels)
#         self.labels = list(self.labels_to_indices.keys())
#         self.length_of_single_pass = self.m_per_class * len(self.labels)
#         self.list_size = length_before_new_iter
#         if self.batch_size is None:
#             if self.length_of_single_pass < self.list_size:
#                 self.list_size -= (self.list_size) % (self.length_of_single_pass)
#         else:
#             assert self.list_size >= self.batch_size
#             assert (
#                     self.length_of_single_pass >= self.batch_size
#             ), "m * (number of unique labels) must be >= batch_size"
#             assert (
#                            self.batch_size % self.m_per_class
#                    ) == 0, "m_per_class must divide batch_size without any remainder"
#             self.list_size -= self.list_size % self.batch_size
#
#     def __len__(self):
#         return self.list_size
#
#     def __iter__(self):
#         idx_list = [0] * self.list_size
#         i = 0
#         num_iters = self.calculate_num_iters()
#         for _ in range(num_iters):
#             c_f.NUMPY_RANDOM.shuffle(self.labels)
#             if self.batch_size is None:
#                 curr_label_set = self.labels
#             else:
#                 curr_label_set = self.labels[: self.batch_size // self.m_per_class]
#             for label in curr_label_set:
#                 t = self.labels_to_indices[label]
#                 idx_list[i: i + self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
#                 i += self.m_per_class
#         return iter(idx_list)
#
#     def calculate_num_iters(self):
#         divisor = (
#             self.length_of_single_pass if self.batch_size is None else self.batch_size
#         )
#         return self.list_size // divisor if divisor < self.list_size else 1
