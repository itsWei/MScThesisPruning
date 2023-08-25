import copy
from typing import Any
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

########################################################################


def gen_mask(row, col, percent=0.5, num_zeros=None):
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int((row * col) * percent)

    mask = np.hstack([np.zeros(num_zeros),
                      np.ones(row * col - num_zeros)])
    np.random.shuffle(mask)
    return mask.reshape(row, col)

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            weight = weight * mask
        
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, weight, bias, mask)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return grad_input, grad_weight, grad_bias, grad_mask
    
class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features
        ))
        
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.output_features)
            )
        else:
            self.register_parameter('bias',None)
            
        self.init_params()
        
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.register_parameter('mask', None)
            
    def init_params(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input):
        return LinearFunction.apply(
            input, self.weight, self.bias, self.mask
        )
    
    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None
        )

class Network(nn.Module):
    def __init__(self, in_size, out_size, ratio=[0,0,0,0]):
        super(Network, self).__init__()
        self.fc1 = CustomizedLinear(in_size, 50, 
                                    mask=gen_mask(in_size, 50, ratio[0]))
        self.bn1 = nn.BatchNorm1d(50)
        
        self.fc2 = CustomizedLinear(50, 50, 
                                    mask=gen_mask(50, 50, ratio[1]))
        self.bn2 = nn.BatchNorm1d(50)
        
        self.fc3 = CustomizedLinear(50, 50, 
                                    mask=gen_mask(50, 50, ratio[2]))
        self.bn3 = nn.BatchNorm1d(50)
        
        self.fc4 = CustomizedLinear(50, out_size, 
                                    mask=gen_mask(50, out_size, ratio[3]))
        self.bn4 = nn.BatchNorm1d(out_size)
        
        self.relu = nn.ReLU()

        # Initialize parameters by following steps:
        # Link: https://pytorch.org/docs/stable/nn.init.html
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0, b=1)
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        return x

########################################################################


class NetworkOptimization(object):
    """docstring for NetworkOptimization"""

    def __init__(self, batch_size, learning_rate, mask_ratio,
                eta= 0.99, data='mnist', edge_changed=1):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.eta = eta
        self.edge_changed = edge_changed
        self.loss_prev = 1e+10  # A whatever big number.
        self.global_step = 0

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # --------------------------------------------------------------
        if data == 'mnist':
            train_dataset = torchvision.datasets.MNIST(
                root='./Datasets/', train=True,
                transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.MNIST(
                root='./Datasets/', train=False,
                transform=transforms.ToTensor(), download=True)
        elif data == 'fashion':
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./Datasets/', train=True,
                transform=transforms.ToTensor(), download=True)
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./Datasets/', train=False,
                transform=transforms.ToTensor(), download=True)

        self.train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        self.test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=10000,
                                                shuffle=True)

        self.model = Network(in_size=784, out_size=10, ratio=mask_ratio)
        self.model = self.model.to(device=self.device)
        self.loss_ckpt = None

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss()

        self.total = len(train_dataset)

    def accuracy(self, outputs, labels):
        pred = torch.argmax(outputs, axis=1) == labels
        pred = pred.cpu().numpy()
        return np.sum(pred) / len(pred)

    def info(self, data, labels, epoch=0):
        # Test the trained model using testing set after each Epoch.
        outputs = self.model(data)
        losses = self.loss_func(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        print('[+] epoch: {0} | test acc: {1} | test loss: {2}'.format(
            epoch, np.round(accuracy, 3), np.round(losses.item(), 3)), end='\r')

        return losses.data.cpu().numpy(), accuracy

    def fit(self, epoch_num, save_step):
        test_data, test_label = next(iter(self.test))
        test_data = test_data.to(device=self.device)
        test_label = test_label.to(device=self.device)
        loss_rec, acc_rec = [], []
        epoch = 0         

        for e in range(epoch_num):

            for batch_img, batch_lab in self.train:
                self.global_step += 1
                batch_img = batch_img.to(device=self.device)
                batch_lab = batch_lab.to(device=self.device)

                outputs = self.model(batch_img.view(self.batch_size, -1))
                loss = self.loss_func(outputs, batch_lab)

                if self.global_step % save_step == 0:
                    acc = self.accuracy(outputs, batch_lab)
                    print('\n[-] loss: ', np.round(loss.item(), 3),
                          '| batch acc: ', np.round(acc, 3), end='\r')

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Test the trained model using testing set after each Epoch.
            loss_cur, acc_cur = self.info(test_data.view(
                test_data.shape[0], -1), test_label, e + 1)
            loss_rec.append(loss_cur)
            acc_rec.append(acc_cur)

            self.loss_prev = loss_cur
            print(acc_cur)

        return loss_rec, acc_rec, epoch_num - epoch

    def link_reduce(self, model, layer, ratio):
        mask_1 = getattr(model, layer[0]).mask.data.clone()
        mask_2 = getattr(model, layer[1]).mask.data.clone()
        mask_3 = getattr(model, layer[2]).mask.data.clone()
        mask_4 = getattr(model, layer[3]).mask.data.clone()
        
        a1 = len(mask_1.flatten())
        a2 = len(mask_2.flatten())
        a3 = len(mask_3.flatten())
        a4 = len(mask_4.flatten())
        
        wanted_links = int((a1+a2+a3+a4) * ratio)

        mask_1_flat = mask_1.flatten()
        mask_2_flat = mask_2.flatten()
        mask_3_flat = mask_3.flatten()
        mask_4_flat = mask_4.flatten()
                
        mask_flat = torch.cat([mask_1_flat, mask_2_flat, mask_3_flat, mask_4_flat])

        # random pruning strategy.  edge 2
        idx = np.arange(len(mask_flat))
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)

        n = 0
        for i in idx:
            if n == wanted_links:
                break

            if mask_flat[i] == 0:
                continue

            mask_flat[i] = 0
            n += 1

        getattr(model, layer[0]).mask.data = mask_flat[0:a1].view(mask_1.shape)
        getattr(model, layer[1]).mask.data = mask_flat[a1:a1+a2].view(mask_2.shape)
        getattr(model, layer[2]).mask.data = mask_flat[a1+a2:a1+a2+a3].view(mask_3.shape)
        getattr(model, layer[3]).mask.data = mask_flat[a1+a2+a3:a1+a2+a3+a4].view(mask_4.shape)
        
        return model

    def link_modify(self, model, layer, num_link):
        for _ in range(num_link):
            mask_1 = getattr(model, layer[0]).mask.data.clone()
            mask_2 = getattr(model, layer[1]).mask.data.clone()
            mask_3 = getattr(model, layer[2]).mask.data.clone()
            mask_4 = getattr(model, layer[3]).mask.data.clone()
            
            a1 = len(mask_1.flatten())
            a2 = len(mask_2.flatten())
            a3 = len(mask_3.flatten())
            a4 = len(mask_4.flatten())
        
            mask_1_flat = mask_1.flatten()
            mask_2_flat = mask_2.flatten()
            mask_3_flat = mask_3.flatten()
            mask_4_flat = mask_4.flatten()
                        
            mask_flat = torch.cat([mask_1_flat, mask_2_flat, mask_3_flat, mask_4_flat])

            # Find out the indices of all active connections.
            idx_T = torch.stack(torch.where(mask_flat == 1), axis=1)
            # Find out the indices of all masked connections.
            idx_F = torch.stack(torch.where(mask_flat == 0), axis=1)

            # Decide how many times of changes we want to go.
            # Randomly generate an integer to pick up 2 numbers.
            T = torch.randint(0, len(idx_T), (1,))[0]
            F = torch.randint(0, len(idx_F), (1,))[0]

            # Disconnect the picked connection ...
            mask_flat[idx_T[T]] = 0
            # ... and connect the disconnected one.
            mask_flat[idx_F[F]] = 1
            
            getattr(model, layer[0]).mask.data = mask_flat[0:a1].view(mask_1.shape)
            getattr(model, layer[1]).mask.data = mask_flat[a1:a1+a2].view(mask_2.shape)
            getattr(model, layer[2]).mask.data = mask_flat[a1+a2:a1+a2+a3].view(mask_3.shape)
            getattr(model, layer[3]).mask.data = mask_flat[a1+a2+a3:a1+a2+a3+a4].view(mask_4.shape)
            
        return model
    
    def Test_Temp(self, layer):
        
        test_data, test_label = next(iter(self.test))
        test_data = test_data.to(device=self.device)
        test_label = test_label.to(device=self.device)
    
        outputs = self.model(test_data.view(test_data.shape[0], -1))
        # loss = self.loss_func(outputs, test_label)
        delta_loss = np.empty(1000)
        for i in range(1000):
            self.model = self.link_modify(self.model, layer,
                                              self.edge_changed)
            
            outputs = self.model(test_data.view(test_data.shape[0], -1))
            loss_new = self.loss_func(outputs, test_label)
            # delta_loss[i] = abs(loss.detach().numpy() - loss_new.detach().numpy())
            delta_loss[i] = loss_new.detach().numpy()
            
        delta = max(delta_loss) - min(delta_loss)
        
        print(delta)
        
        return delta

    def mask_ratio(self, layer):
        msk_1 = getattr(self.model, layer[0]).mask.data.cpu().numpy()
        msk_2 = getattr(self.model, layer[1]).mask.data.cpu().numpy()
        msk_3 = getattr(self.model, layer[2]).mask.data.cpu().numpy()
        msk_4 = getattr(self.model, layer[3]).mask.data.cpu().numpy()
        
        a = np.sum(msk_1) / len(msk_1.flatten())
        b = np.sum(msk_2) / len(msk_2.flatten())
        c = np.sum(msk_3) / len(msk_3.flatten())
        d = np.sum(msk_4) / len(msk_4.flatten())
                
        return a,b,c,d

    def FCSA(self, layer, reduce, metropolis,iteration):
        test_data, test_label = next(iter(self.test))
        test_data = test_data.to(device=self.device)
        test_label = test_label.to(device=self.device)

        self.model = self.link_reduce(self.model, layer, reduce)
        outputs = self.model(test_data.view(test_data.shape[0], -1))
        loss = self.loss_func(outputs, test_label)
        T = self.Test_Temp(layer)
        
        loss_iter = []
        
        for i in np.arange(iteration):
            for m in range(metropolis):
                mask_1_pre = getattr(self.model, layer[0]).mask.data.clone()
                mask_2_pre = getattr(self.model, layer[1]).mask.data.clone()
                mask_3_pre = getattr(self.model, layer[2]).mask.data.clone()
                mask_4_pre = getattr(self.model, layer[3]).mask.data.clone()
                
                self.model = self.link_modify(self.model, layer,
                                              self.edge_changed)

                outputs = self.model(test_data.view(test_data.shape[0], -1))
                loss_mdf = self.loss_func(outputs, test_label)
                delta = loss_mdf.data.cpu() - loss.data.cpu()  # prev > curr

                r = np.random.random()
                if (delta <= 0):
                    loss = loss_mdf
                elif (r < torch.exp((-delta)/(T))):
                    loss = loss_mdf
                else:
                    getattr(self.model, layer[0]).mask.data = mask_1_pre
                    getattr(self.model, layer[1]).mask.data = mask_2_pre
                    getattr(self.model, layer[2]).mask.data = mask_3_pre
                    getattr(self.model, layer[3]).mask.data = mask_4_pre
                    
                loss_iter.append(loss.data.cpu())
            # Temperature decrease at each epoch.
            T *= self.eta
        
        msk_1, msk_2, msk_3, msk_4 = self.mask_ratio(layer)
        # Test the trained model using testing set.
        outputs = self.model(test_data.view(test_data.shape[0], -1))
        losses = self.loss_func(outputs, test_label)
        accuracy = self.accuracy(outputs, test_label)
        print('[+] round: {0} | metro: {1} | test acc: {2} | test loss: {3} | T: {4} | M1: {5}% | M2: {6}% | M3: {7}% | M4: {8}% '.format(
            i + 1, m + 1,
            np.round(accuracy, 3),
            np.round(losses.item(), 3),
            np.round(T, 3),
            np.round(msk_1 * 100, 3), 
            np.round(msk_2 * 100, 3), 
            np.round(msk_3 * 100, 3), 
            np.round(msk_4 * 100, 3)), end='\r')
                
        loss_metro = np.round(losses.item(), 5)
        acc_metro = np.round(accuracy, 5)

        return loss_metro, acc_metro, loss_iter