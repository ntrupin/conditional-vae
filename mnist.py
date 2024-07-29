# adapted from https://github.com/hsjeong5/MNIST-for-Numpy

import gzip
import os
import pickle
from urllib import request

import numpy as np

FILES = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]

def download_mnist(base_url, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for name in FILES:
        print(f"Downloading {name[1]}...")
        request.urlretrieve(base_url + name[1], os.path.join(save_dir, name[1]))
    print("Download complete.")

def save_mnist(save_dir, filename):
    mnist = {}
    for name in FILES[:2]:
        path = os.path.join(save_dir, name[1])
        with gzip.open(path, "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16)\
                .reshape(-1, 28 * 28)
    for name in FILES[-2:]:
        path = os.path.join(save_dir, name[1])
        with gzip.open(path, "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def mnist(
    save_dir="./tmp",
    base_url="https://raw.githubusercontent.com/fgnt/mnist/master/",
    filename="mnist.pkl",
):
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        download_mnist(base_url, save_dir)
        save_mnist(save_dir, filename)
    with open(path, "rb") as f:
        mnist = pickle.load(f)

    def preproc(x):
        return x.astype(np.float32) / 255.0

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["test_images"] = preproc(mnist["test_images"])
    return (
        mnist["training_images"],
        mnist["training_labels"].astype(np.uint32),
        mnist["test_images"],
        mnist["test_labels"].astype(np.uint32),
    )
