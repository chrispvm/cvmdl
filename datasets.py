import os

import matplotlib.pyplot as plt

current_directory = os.getcwd()


# def load_mnist(batch_size_train, batch_size_test, return_datasets=False):
#     return get_data_loader(load_mnist_dataset(), batch_size_train, batch_size_test, return_datasets, 60000, 10000)


def print_six_mnist_images(loader):
    sampler = enumerate(loader)
    batch_nr, (batch_images, batch_labels) = next(sampler)
    fig = plt.figure()
    for i in range(6):
        print(batch_images[i].shape)
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(batch_images[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(batch_labels[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()
