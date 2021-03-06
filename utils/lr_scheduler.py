##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, bad_count):
        if self.mode == 'adam':
            if epoch < 20:
                lr = 0.01
            elif epoch >= 20 and epoch < 30:
                lr = 0.001
            elif epoch >= 30:
                lr = 0.0001
        elif self.mode == 'custom':
            if bad_count >= 2:
                self.lr *= 0.5
            lr = self.lr
        elif self.mode == 'cyclic':
            i = epoch % 10
            i = i // 2
            if i == 0 or i == 4:
                lr = 1e-6
            elif i == 1 or i == 3:
                lr = 1e-5
            else:
                lr = 1e-4
            self.lr = lr
        else :
            T = epoch * self.iters_per_epoch + i
            if self.mode == 'cos':
                lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            elif self.mode == 'poly':
                lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
            elif self.mode == 'step':
                lr = self.lr * (0.1 ** (epoch // self.lr_step))
            else:
                raise NotImplemented
            # warm up lr schedule
            if self.warmup_iters > 0 and T < self.warmup_iters:
                lr = lr * 1.0 * T / self.warmup_iters

        if epoch > self.epoch:
            # print('\n=>Epoches %i, learning rate = %.3E, \
            #     previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) > 1:
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:
                    param_group['lr'] = lr * 0.1
                else:
                    param_group['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
