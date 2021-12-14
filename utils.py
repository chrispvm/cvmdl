import inspect
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch.nn as nn


current_directory = os.getcwd()
print('current directory: ', current_directory)


def yes_no_to_bool(x):
    assert type(x) is str
    if x == 'y' or x == 'Y':
        return True
    if x == 'n' or x == 'N':
        return False
    raise ValueError('Needs to be Y or N')


def make_dir(my_path):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(my_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else:
            raise


def plt_savefig(file_name, dir_path, sub_folder=None):
    make_dir(dir_path)
    if sub_folder:
        dir_path = f"{dir_path}/{sub_folder}"
        make_dir(dir_path)
    plt.savefig(f"{dir_path}/{file_name}.png")


def moving_average(values, window):
    if window > len(values):
        return values
    else:
        weights = np.repeat(1.0, window) / window
        values = np.pad(values, (window // 2, window -
                        1 - window // 2), mode='edge')
        smas = np.convolve(values, weights, 'valid')
        # smas = np.convolve(values, weights, 'same')
        return list(smas)


def has_children(module: nn.Module):
    has = False
    for c in module.children():
        has = True
    return has


def named_flatten_module(module: nn.Module):
    named_modules = []
    for (name, m) in module.named_modules():
        if not has_children(m):
            named_modules += [(name, m)]
    if not named_modules:
        named_modules = [("root", module)]
    return named_modules


def flatten_module(module: nn.Module):
    return [m for (name, m) in named_flatten_module(module)]


def _flatten_list(my_list):
    for i in my_list:
        if isinstance(i, (list, tuple)):
            for j in _flatten_list(i):
                yield j
        else:
            yield i


def flatten_list(my_list):
    return list(_flatten_list(my_list))


def plot_image_and_classification_probabilities(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def get_pd_index_dims(index: pd.Index):
    if isinstance(index, pd.MultiIndex):
        return [len(level) for level in index.levels]
    elif isinstance(index, pd.Index):
        return [len(index)]
    else:
        print("index", index)
        breakpoint()
        raise NotImplementedError


def dataframe_to_mdarray(df: pd.DataFrame):
    index_shape = get_pd_index_dims(df.index)
    column_shape = get_pd_index_dims(df.columns)
    #     print("shapes: ", column_shape, " and: ", index_shape)
    return df.values.reshape(index_shape + column_shape)
