import numpy as np
from typing import List
from itertools import product

class Linear:
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def forward(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        self.input = input
        self.output = input @ self.weight.T + self.bias if self.bias is not None else input @ self.weight.T
        return self.output

    def backward(self, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        return grad_output @ self.weight

    def update_grad_parameters(self, grad_output: np.array):
        """
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += grad_output.T @ input
        if self.grad_bias is not None:
            self.grad_bias += grad_output.sum(axis=0)


    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = (padding if isinstance(padding, tuple) else (padding, padding))
        self._init_weights(in_channels, out_channels, self.kernel_size)

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 1 / np.sqrt(in_channels * kernel_size[0] * kernel_size[1])

        self.weight = {
            "W": np.random.normal(loc=0, scale=scale, size=(out_channels, in_channels, *kernel_size)),
            "b": np.zeros(shape=(out_channels, 1)),
        }
        self.weight_update = {'W': None, 'b': None}
    def forward(self, input):
        """
        Forward pass for the convolution layer.
        Args:
            X: numpy.ndarray of shape (N, C, H_in, W_in).
        Returns:
            Y: numpy.ndarray of shape (N, F, H_out, W_out).
        """
        self.input = input
        if self.padding:
            self.input = np.pad(self.input, pad_width=((0, 0), (0, 0), self.padding, self.padding))
        
        N, C, H, W = self.input.shape
        KH, KW = self.kernel_size
        out_shape = (N, self.out_channels, 1 + (H - KH) // self.stride, 1 + (W - KW) // self.stride)
        Y = np.zeros(out_shape)
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    rec_field = self.input[n, :, h_offset : h_offset + KH, w_offset : w_offset + KW]
                    Y[n, c_w, h, w] = (
                        np.sum(self.weight["W"][c_w] * rec_field) + self.weight["b"][c_w]
                    )

        return Y

    def backward(self, dY):
        
        dX = np.zeros_like(self.input)
        N, C, H, W = dX.shape
        KH, KW = self.kernel_size
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(dY.shape[2]), range(dY.shape[3])):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    dX[n, :, h_offset : h_offset + KH, w_offset : w_offset + KW] += (
                        self.weight["W"][c_w] * dY[n, c_w, h, w]
                    )

        # calculating the global gradient wrt the conv filter weights
        dW = np.zeros_like(self.weight["W"])
        for c_w in range(self.out_channels):
            for c_i in range(self.in_channels):
                for h, w in product(range(KH), range(KW)):
                    X_rec_field = self.input[
                        :, c_i, h : H - KH + h + 1 : self.stride, w : W - KW + w + 1 : self.stride
                    ]
                    dY_rec_field = dY[:, c_w]
                    dW[c_w, c_i, h, w] = np.sum(X_rec_field * dY_rec_field)

        # calculating the global gradient wrt to the bias
        db = np.sum(dY, axis=(0, 2, 3)).reshape(-1, 1)

        # caching the global gradients of the parameters
        self.weight_update["W"] = dW
        self.weight_update["b"] = db

        return dX


class BaseBatchNorm:
    """
    Applies batch normalization transformation
    """

    def __init__(self):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
      
        self.axis = None
        self.index = None
        self.final_index = None
        self.shape = None
        self.reshape = None
        self.training = True
    
    def forward(self, input):
        x_reshaped = input.reshape(self.reshape)  # (WxH) or (N, C, -1)
        self.mean = np.mean(x_reshaped, axis=self.axis)
        self.var = np.var(x_reshaped, axis=self.axis)

        if self.training:
            self.norm_input = (x_reshaped - self.mean[self.index]) / np.sqrt(self.var[self.index] + self.eps)
            self.norm_input = self.norm_input.reshape(self.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.norm_input = (input - self.running_mean[self.final_index]) / np.sqrt(self.running_var[self.final_index] + self.eps)

        output = self.norm_input * self.weight[self.final_index] + self.bias[self.final_index] if self.affine else self.norm_input
        return output
    
    def backward(self, grad_output):
        if not self.training:
            grad_input = grad_output / np.sqrt(self.running_var[self.final_index] + self.eps)
            return grad_input * self.weight[self.final_index] if self.affine else grad_input

        grad_output_reshaped = grad_output.reshape(self.reshape)
        grad_mean = np.mean(grad_output_reshaped, axis=self.axis)
        grad_var = np.var(grad_output_reshaped, axis=self.axis)

        x_normalized = (grad_output_reshaped - grad_mean[self.index]) / (np.sqrt(grad_var[self.index] + self.eps))
        x_normalized = x_normalized.reshape(self.shape)
        grad_input = self.weight[self.final_index] * x_normalized if self.affine else x_normalized
        return grad_input


class MyBatchNorm1d(BaseBatchNorm):
    """
    :param num_features:
    :param eps:
    :param momentum:
    :param affine: whether to use trainable affine parameters
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool =True):
        super(MyBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None
    
    def forward(self, input):
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """

        self.axis = 0
        self.shape = input.shape
        self.index = np.arange(self.shape[1])
        self.final_index = np.arange(self.shape[1])
        self.reshape = input.shape
        return super().forward(input)
    
    def backward(self, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        return super().backward(grad_output)
    def training(self):
        self.training = True

    def eval(self):
        self.training = False





class MyBatchNorm2d(BaseBatchNorm):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool =True):
        super(MyBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None
    
    def forward(self, input):
        """
        :param input: array of shape (batch_size, num_features, WxH)
        :return: array of shape (batch_size, num_features, WxH)
        """
        N, C, H, W = input.shape
        self.axis = 0, 2
        self.shape = N, C, H, W
        self.index = (None, np.arange(self.shape[1]), None)
        self.final_index = (None, np.arange(self.shape[1]), None, None)
        self.reshape = N, C, -1
        return super().forward(input)
    
    def backward(self, grad_output: np.array) -> np.array:
        """
        :param grad_output: array of shape (batch_size, num_features, WxH)
        :return: array of shape (batch_size, num_features, WxH)
        """
        return super().backward(grad_output)
    

class Dropout:
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def forward(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if not self.training:
            return input
        self.mask = np.random.binomial(n=1, p=1-self.p, size=input.shape)
        output = 1/(1-self.p) * self.mask * input
        return output

    def backward(self, grad_output: np.array) -> np.array:
        """
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if not self.training:
            return grad_output
        return 1/(1-self.p) * self.mask * grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential:
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def forward(self, input: np.array) -> np.array:
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})

        Just write a little loop.

        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """

        self.input = input
        for module in self.modules:
            self.output = module(self.input)

        return self.output

    def backward(self, grad_output: np.array) -> np.array:
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, output_grad)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            grad_input = module[0].backward(input, g_1)

        !!!
        To each module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.
        !!!

        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """

        l = len(self.modules)
        grad_input = grad_output
        for ind, module in enumerate(self.modules[::-1]):
            inp = self.modules[-ind - 2].output if ind < l - 1 else self.input
            grad_input = module.backward(inp, grad_input)
        return grad_input

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
