import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


import torch
from torch.optim import Optimizer
import math



###########################################################################

# Impplementation SGD and Adam on Numpy

###########################################################################

class Optimizer(ABC):
    """
    Basic class for all optimizers
    """
    def __init__(self):
        """
        :param module: neural network containing parameters to optimize
        """
        self.state = {}  # storing current state of optimizer

    def zero_grad(self):
        """
        Zero module gradients
        """
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        """
        Process one step of optimizer
        """
        raise NotImplementedError
    

class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
            """
              - update momentum variable (m)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
            grad += self.weight_decay * param
            np.add(self.momentum * m, grad, out=m)
            grad = m
            np.add(param, -self.lr*grad, out=param)
        return parameters


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            """
              - update first moment variable (m)
              - update second moment variable (v)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
            grad += self.weight_decay * param
            np.add(self.beta1*m, (1-self.beta1)*grad, out=m)
            np.add(self.beta2*v, (1-self.beta2)*grad**2, out=v)
            m_hat = 1/(1-self.beta1**self.state['t']) * m
            v_hat = 1/(1-self.beta2**self.state['t']) * v
            np.add(param, -self.lr * m_hat/(np.sqrt(v_hat)+self.eps), out=param)
        return parameters


###########################################################################

# Impplementation Nesterov, RMSprop and Adam on PyTorch

###########################################################################


class NesterovOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum) 
        super(NesterovOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                speed = state['momentum_buffer']
                speed.mul_(momentum).add_(grad)
                p.data.add_(-lr, grad + momentum * speed)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.9, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for params in group['params']:
                if params.grad is None:
                    continue
                grad = params.grad.data
                state = self.state[params]

                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(params.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                denom = square_avg.sqrt().add_(group['eps'])
                params.data.addcdiv_(-group['lr'], grad, denom)

        return loss


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for params in group['params']:
                if params.grad is None:
                    continue
                grad = params.grad.data
                state = self.state[params]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(params.data)
                    state['exp_avg_sq'] = torch.zeros_like(params.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = (exp_avg_sq.sqrt() / (math.sqrt(bias_correction2) + group['eps']))
                step_size = group['lr'] / bias_correction1

                params.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    params.data.add_(-group['lr'] * group['weight_decay'], params.data)

        return loss