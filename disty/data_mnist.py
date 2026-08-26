"""
data_mnist.py - load the MNIST subset that ships with this folder.

WHY A FILE IN THE REPO INSTEAD OF A DOWNLOAD
Every node must train on exactly the same dataset, and a demo should not depend
on the network being up. mnist_subset.npz holds 10,000 training rows and 2,000
test rows taken from the real MNIST set, 1,000 and 200 per digit so the classes
are balanced, chosen once with a fixed seed. It is 2 MB, so it travels with the
repository and every machine is guaranteed to have byte-identical data.

WHAT IS IN THE FILE
    train_x  uint8  (10000, 784)    28x28 pixels flattened, 0-255
    train_y  uint8  (10000,)        the digit, 0-9
    test_x   uint8  (2000, 784)
    test_y   uint8  (2000,)

The images are flattened already because the model here is a plain
fully-connected network, so it never needs to know the images were square.
"""

import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET = os.path.join(HERE, "mnist_subset.npz")


def load(kind="train"):
    """
    Return (inputs, targets) for "train" or "test".

    inputs  is float64, one row per image, every pixel scaled to 0.0-1.0
    targets is int64, one label per row

    float64 to match the model - see build_model in train_mnist.py for why the
    whole thing runs in double precision rather than the usual float32.

    Dividing by 255 is the only preprocessing. Without it the inputs are 0-255
    and the first layer's outputs are huge, which makes the very first steps
    unstable for no reason.
    """
    if not os.path.exists(SUBSET):
        raise FileNotFoundError(
            f"{SUBSET} is missing. It is meant to be committed alongside this "
            f"file - check the repository rather than trying to download it.")

    data = np.load(SUBSET)
    images = data[f"{kind}_x"]
    labels = data[f"{kind}_y"]

    inputs = torch.tensor(images, dtype=torch.float64) / 255.0
    targets = torch.tensor(labels, dtype=torch.int64)
    return inputs, targets
