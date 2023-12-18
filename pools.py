import numpy as np
import torch
import torch.nn as nn
from itertools import product


class MaxPool2D:
    def __init__(self, kernel_size=(2, 2), stride=2, padding=0):
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = padding

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        Y = np.zeros((N, C, H // KH, W // KW))

        # for n in range(N):
        for h, w in product(range(0, H // KH), range(0, W // KW)):
            h_offset, w_offset = h * KH, w * KW
            rec_field = X[:, :, h_offset : h_offset + KH, w_offset : w_offset + KW]
            Y[:, :, h, w] = np.max(rec_field, axis=(2, 3))
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset + kh, w_offset + kw] = (
                    X[:, :, h_offset + kh, w_offset + kw] >= Y[:, :, h, w]
                )

        self.grad = grad
        return Y

    def backward(self, dY):
        dY = np.repeat(
            np.repeat(dY, repeats=self.kernel_size[0], axis=2), repeats=self.kernel_size[1], axis=3
        )
        return self.grad * dY

    def local_grad(self, X):
        return self.grad



class AvgPool2D:
    def __init__(self, kernel_size=(2, 2), stride=2, padding=0):
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = ((padding, padding) if isinstance(padding, int) else padding)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        Y = np.zeros((N, C, (H) // KH, (W) // KW))

        for h, w in product(range(0, Y.shape[1] // KH), range(0, Y.shape[3] // KW)):
            h_offset, w_offset = h * KH, w * KW
            rec_field = X[:, :, h_offset : h_offset + KH, w_offset : w_offset + KW]
            Y[:, :, h, w] = np.mean(rec_field, axis=(2, 3))
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset + kh, w_offset + kw] = (
                    X[:, :, h_offset + kh, w_offset + kw] >= Y[:, :, h, w]
                )

        self.grad = grad
        return Y

    def backward(self, dY):
        dY = np.repeat(
            np.repeat(dY, repeats=self.kernel_size[0], axis=2), repeats=self.kernel_size[1], axis=3
        )
        return self.grad * dY

    def local_grad(self, X):
        return self.grad