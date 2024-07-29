from dataclasses import dataclass
import numpy as np
import os
import time
import torch
from torch import nn
import torchvision
from typing import Tuple

from mnist import mnist
from cvae import CVAE, CVAEEmbedder

device = torch.device("mps")

@dataclass
class ModelArgs:
    batch_size: int = 128
    image_size: Tuple[int, ...] = (1, 64, 64)
    num_filters: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    latent_dim: int = 8

def batch_iterate(batch_size, data, labels):
    perm = torch.randperm(data.shape[0])
    for i, s in enumerate(range(0, data.shape[0], batch_size)):
        ids = perm[s:s+batch_size]
        yield i, data[ids], labels[ids]

def vae_loss(x, y, mu, logvar):
    # reconstruction loss
    recon_loss = nn.functional.mse_loss(y, x, reduction="sum")

    # kl divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())

    return recon_loss + kld

def pytorchify(images):
    """
    mnist gives us 28x28 images, which my old dataloader gives
    us a 784 element vector. our net expects 1x64x64 tensors,
    so this function does that.
    """
    resize = torchvision.transforms.Resize((64, 64))
    images = (torch.tensor(images[:, np.newaxis, ...])
        .reshape((-1, 1, 28, 28)))
    return resize(images)

def train(args):
    rsz = torchvision.transforms.Resize((64, 64))
    train_images, train_labels, test_images, _ = mnist()
    train_images, test_images = pytorchify(train_images).to(device), pytorchify(test_images)

    # pytorch doesn't support upsampling convolutions on an mps backend.
    embedder = CondVAEEmbedder() 
    model = CondVAE(args.image_size, args.latent_dim, 10, args.num_filters)
    model.to(device)

    train_labels = embedder.encode([f"{x}" for x in train_labels], device)

    optimizer = torch.optim.AdamW(model.parameters(),
        lr=args.learning_rate)

    for e in range(1, args.epochs + 1):
        model.train()

        tic = time.perf_counter()
        loss_acc = 0.0
        throughput_acc = 0.0

        for i, batch, labels in batch_iterate(args.batch_size, train_images, train_labels):
            throughput_tic = time.perf_counter()

            y, mu, logvar = model(batch, labels)
            loss = vae_loss(batch, y, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            throughput_toc = time.perf_counter()
            throughput_acc += batch.shape[0] / (throughput_toc - throughput_tic)
            loss_acc += loss.item()

            if i > 0 and (i % 10 == 0):
                print(" | ".join([
                    f"Epoch {e:4d}",
                    f"Loss {(loss_acc / i):10.2f}",
                    f"Throughput {(throughput_acc / i):8.2f} im/s",
                    f"Batch {i:5d}"
                    ]), end="\r")

        toc = time.perf_counter()

        print(" | ".join([
            f"Epoch {e:4d}",
            f"Loss {(loss_acc / i):10.2f}",
            f"Throughput {(throughput_acc / i):8.2f} im/s",
            f"Time {toc - tic:8.1f} (s)"
            ]))

        model.eval()

        torch.save(model.state_dict(), "./cvae.pth")


if __name__ == "__main__":
    args = ModelArgs()
    if os.path.isfile("cvae.pth"):
        embedder = CVAEEmbedder()
        model = CVAE(args.image_size, args.latent_dim, 10, args.num_filters)
        model.load_state_dict(torch.load("cvae.pth"))
        model.to(device)
        cmd = ""
        while True:
            cmd = input("> ")
            if cmd == "exit": break
            embedding = embedder.encode([cmd], device)
            image = model.generate(embedding, device)
            torchvision.utils.save_image(image, "demo.png")
    else:
        train(args)
