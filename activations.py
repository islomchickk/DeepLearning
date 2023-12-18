import numpy as np
import torch
import torch.nn as nn


class ReLU:
    """
    Applies element-wise ReLU function
    """
    def __call__(self, input):
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return self.forward(input)
    
    def forward(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def backward(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_output[input<=0] = 0
        return grad_output


class MySigmoid:
    """
    Applies element-wise sigmoid function
    """
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.input = input
        return 1/(1+np.exp(- self.input))

    def backward(self, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return np.exp(-self.input) / (1+np.exp(-self.input))**2 * grad_output
    

class MyTanh:
    """
    Applies element-wise tanh function
    """
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.out = np.tanh(input)
        return self.out

    def backward(self, grad_output: np.array) -> np.array:
        """
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (1 - self.out**2)


