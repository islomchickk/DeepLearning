import numpy as np


class MSELoss:
    """
    Mean squared error criterion
    """
    def __call__(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return self.forward(input, target)
    
    def forward(self, input: np.array, target: np.array) -> float:
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `_compute_output`.

        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        # print(input.shape, target.shape)
        assert input.shape == target.shape, 'input and target shapes not matching'
        m, n = input.shape
        self.input = input
        self.target = target
        self.output = ((self.input - self.target) ** 2).sum() / (m * n)
        return self.output

    def backward(self) -> np.array:
        """
        :return: array of size (batch_size, *)
        """
        assert self.input.shape == self.target.shape, 'input and target shapes not matching'
        grad_MSE = (2 / self.input.size * (self.input- self.target)).sum()
        return grad_MSE


def np_logsumexp(data, axis=-1):
    assert (data.shape[1] > axis and data.shape[0] > axis), f'Please choose true axis! axis must be lower than {min(data.shape)}'
    
    m = np.max(data)
    res = m + np.log(np.sum(np.exp(data-m), axis=axis))
    return res


class MySoftmax:
    """
    Applies Softmax operator over the last dimension
    """

    def __call__(self, input: np.array) -> float:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return self.forward(input)

    def forward(self, input: np.array) -> float:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.input = input
        exp_inputs = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
        softmax_probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return softmax_probs

    def backward(self, grad_output: np.array) -> np.array:
        """
        :param grad_output: array of the same size
        :return: array of the same size
        """
        m, n = self.input.shape
        sftm = np.exp(self.input) / sum(sum(np.exp(self.input)))
        tensor_1 = np.einsum("ij,ik->ijk", sftm, sftm)
        tensor_2 = np.einsum("ij,jk->ijk", sftm, np.eye(n, n))
        return np.einsum('ijk, ik->ij', tensor_2 - tensor_1, grad_output)
    


class LogSoftmax:
    """
    Applies LogSoftmax operator over the last dimension
    """
    def __init__(self):
        self.softmax = MySoftmax()

    def forward(self, input: np.array) -> np.array:
        m = input.shape[0]
        self.input = input
        return self.input - np_logsumexp(self.input, axis=1).reshape((m, 1))
    
    def backward(self, grad_output: np.array) -> np.array:
        """
        :param grad_output: array of the same size
        :return: array of the same size
        """
        m, n = self.input.shape
        sftmx = self.softmax.forward(self.input)

        tensor1 = np.einsum('ij,ik->ijk', sftmx, np.ones((m, n)))
        tensor2 = np.einsum('ij,jk->ijk', np.ones((m, n)), np.eye(n, n))

        return np.einsum('ijk, ik->ij', tensor2 - tensor1, grad_output)
    

class MyCrossEntropyLoss:
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.0):
        self.weights = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softmax = MySoftmax()
        self.log_softmax = LogSoftmax()

    def __call__(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        return self.forward(input, target)
    
    def forward(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        m, n = input.shape
        self.input = input
        self.target = target

        log_p = self.log_softmax.forward(self.input)[range(m), self.target]
        if self.weights is not None:
            if self.reduction == 'mean':
                return -(log_p * self.weights[self.target] / self.weights[self.target].sum()).sum()
            elif self.reduction == 'sum':
                return -(log_p * self.weights[self.target]).sum()
        if self.reduction == 'mean':
            return -log_p.sum() / m
        return -log_p.sum()

    def backward(self) -> np.array:
        """
        :return: array of size (batch_size, num_classes)
        """
        m, n = self.input.shape
        prob_inputs = self.softmax.forward(self.input)
        prob_inputs[np.arange(m), self.target] -= 1.0
        if self.weights is not None:
            return (prob_inputs * self.weights[self.target].reshape(-1, 1))
        if self.reduction == 'mean':
            return prob_inputs / m
        return prob_inputs