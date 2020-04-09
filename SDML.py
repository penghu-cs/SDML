import torch
import torch.nn as nn
import numpy as np

from torch import optim
import utils_PyTorch as utils
import torch.nn.functional as F
import data_loader
from torch.autograd import Variable
import scipy.io as sio
import copy
import math
import multiprocessing
import time

class Solver(object):
    def __init__(self, config):
        wv_matrix = None

        self.output_shape = config.output_shape
        data = data_loader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        (self.train_data, self.train_labels, self.valid_data, self.valid_labels, self.test_data, self.test_labels, self.MAP) = data

        self.n_view = len(self.train_data)
        for v in range(self.n_view):
            if min(self.train_labels[v].shape) == 1:
                self.train_labels[v] = self.train_labels[v].reshape([-1])
            if min(self.valid_labels[v].shape) == 1:
                self.valid_labels[v] = self.valid_labels[v].reshape([-1])
            if min(self.test_labels[v].shape) == 1:
                self.test_labels[v] = self.test_labels[v].reshape([-1])

        if len(self.train_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.train_labels).reshape([-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.train_labels[0].shape[1]

        if self.output_shape == -1:
            self.output_shape = self.num_classes

        self.word_dim = 300
        self.dropout_prob = 0.5
        if wv_matrix is not None:
            self.vocab_size = wv_matrix.shape[0] - 2

        self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.view_id = config.view_id

        self.epochs = config.epochs
        self.sample_interval = config.sample_interval

        self.compute_all = config.compute_all
        self.just_valid = config.just_valid
        self.multiprocessing = config.multiprocessing

        if self.batch_size < 0:
            self.batch_size = 100 if self.num_classes < 100 else 500

        W_FileName = 'OrthP_' + str(self.output_shape) + 'X' + str(self.output_shape) + '.mat'
        try:
            self.W = sio.loadmat(W_FileName)['W']
        except Exception as e:
            W = torch.Tensor(self.output_shape, self.output_shape)
            W = torch.nn.init.orthogonal(W, gain=1)
            self.W = self.to_data(W)
            sio.savemat(W_FileName, {'W': self.W})
        self.W = self.W[:, 0: self.num_classes]
        self.runing_time = config.running_time

        if self.just_valid:
            self.tr_d_loss = multiprocessing.Manager().list([[] for i in range(self.n_view)])
            self.val_d_loss = multiprocessing.Manager().list([[] for i in range(self.n_view)])
            self.tr_ae_loss = multiprocessing.Manager().list([[] for i in range(self.n_view)])
            self.val_ae_loss = multiprocessing.Manager().list([[] for i in range(self.n_view)])

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x) #torch.autograd.Variable

    def to_data(self, x):
        """Converts variable to numpy."""
        try:
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy()
        except Exception as e:
            return x

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        # self.i_optimizer.zero_grad()

    def to_one_hot(self, x):
        if len(x.shape) == 1 or x.shape[1] == 1:
            one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
            labels = one_hot
            y = self.to_var(torch.tensor(labels))
        else:
            y = self.to_var(torch.tensor(x.astype('float32')))
        return y

    def view_result(self, _acc):
        res = ''
        if type(_acc) is not list:
            res += ((' - mean: %.5f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
            for _i in range(self.n_view):
                for _j in range(self.n_view):
                    if _i != _j:
                        res += ('%.5f' % _acc[_i, _j]) + ','
        else:
            R = [50, 'ALL']
            for _k in range(len(_acc)):
                res += (' R = ' + str(R[_k]) + ': ')
                res += ((' - mean: %.5f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.5f' % _acc[_k][_i, _j]) + ','
        return res

    def shuffleTrain(self):
        for v in range(self.n_view):
            for ci in range(self.num_classes):
                inx = np.arange(len(self.train_view_list[1][v][ci]))
                np.random.shuffle(inx)
                self.train_view_list[0][v][ci] = self.train_view_list[0][v][ci][inx]
                self.train_view_list[1][v][ci] = self.train_view_list[1][v][ci][inx]
            self.train_data[v] = np.concatenate(self.train_view_list[0][v])
            self.train_labels[v] = np.concatenate(self.train_view_list[1][v])

    def train(self):
        if self.multiprocessing and self.view_id < 0:
            import torch.multiprocessing as mp
            mp = mp.get_context('spawn')
            process = []
            start = time.time()
            for v in range(self.n_view):
                process.append(mp.Process(target=self.train_view, args=(v,)))
                process[v].daemon = True
                process[v].start()
            for p in process:
                p.join()
        elif self.view_id >= 0:
            start = time.time()
            self.train_view(self.view_id)
        else:
            start = time.time()
            for v in range(self.n_view):
                self.train_view(v)
        end = time.time()
        runing_time = end - start
        if self.runing_time:
            print('runing_time: ' + str(runing_time))
            return runing_time

        if not self.just_valid:
            # valid = [self.resutls[v][0] for v in range(self.n_view)]
            # test = [self.resutls[v][1] for v in range(self.n_view)]
            valid_fea, valid_lab, test_fea, test_lab = [], [], [], []
            for v in range(self.n_view):
                tmp = sio.loadmat('features/' + self.datasets + '_' + str(v) + '.mat')
                valid_fea.append(tmp['valid_fea'])
                valid_lab.append(tmp['valid_lab'].reshape([-1,]) if min(tmp['valid_lab'].shape) == 1 else tmp['valid_lab'])
                test_fea.append(tmp['test_fea'])
                test_lab.append(tmp['test_lab'].reshape([-1,]) if min(tmp['test_lab'].shape) == 1 else tmp['test_lab'])

            valid_results = utils.multi_test(valid_fea, valid_lab, self.MAP)
            test_results = utils.multi_test(test_fea, test_lab, self.MAP)
            print("valid results: " + self.view_result(valid_results) + ",\t test resutls:" + self.view_result(test_results))
            sio.savemat('features/' + self.datasets + '_SDML_test_feature_results.mat', {'test': test_fea, 'test_labels': test_lab})


            return valid_results, test_results
        else:
            return np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.val_d_loss], axis=0), np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.tr_d_loss], axis=0), np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.tr_ae_loss], axis=0)
        # print("best resutls:" + self.view_result(best_test_results))
        # return best_test_results

    def train_view(self, view_id):
        seed = 0
        import numpy as np
        np.random.seed(seed)
        import random as rn
        rn.seed(seed)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(view_id % 2)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        from model import Dense_Net
        Net = Dense_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape)
        AE = Dense_Net(input_dim=self.output_shape, out_dim=self.input_shape[view_id])

        if torch.cuda.is_available():
            Net.cuda()
            AE.cuda()

        W = torch.tensor(self.W)
        W = Variable(W.cuda(), requires_grad=False)
        get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
        params = get_grad_params(Net) + get_grad_params(AE)
        optimizer = optim.Adam(params, self.lr[view_id], [self.beta1, self.beta2])

        discriminator_losses, losses, valid_results = [], [], []
        # criterion = lambda x, y: ((x - y) ** 2).sum()
        criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()
        tr_d_loss, tr_ae_loss, val_d_loss, val_ae_loss = [], [], [], []
        # criterion = lambda x, y: (((x - y) ** 2).sum(1) / 2.).mean()
        valid_loss_min = 1e9
        for epoch in range(self.epochs):
            print(('ViewID: %d, Epoch %d/%d') % (view_id, epoch + 1, self.epochs))

            rand_idx = np.arange(self.train_data[view_id].shape[0])
            np.random.shuffle(rand_idx)
            # batch_count = int(np.ceil(self.train_data[view_id].shape[0] / float(self.batch_size)))
            batch_count = int(self.train_data[view_id].shape[0] / float(self.batch_size))

            k = 0
            mean_loss = []
            mean_tr_d_loss, mean_tr_ae_loss = [], []
            for batch_idx in range(batch_count):
                idx = rand_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                train_y = self.to_one_hot(self.train_labels[view_id][idx])
                train_x = self.to_var(torch.tensor(self.train_data[view_id][idx]))

                optimizer.zero_grad()
                out_net = Net(train_x)[-1]
                pred = out_net.view([out_net.shape[0], -1]).mm(W)
                # pred = Net(train_x)[-1]
                ae_data = train_x
                ae_pred = AE(out_net)[-1]
                ae_loss = criterion(ae_pred, ae_data) * self.alpha
                labeled_inx = train_y.sum(1) > 0
                d_loss = criterion(pred[labeled_inx], train_y[labeled_inx]) * (1 - self.alpha)
                loss = ae_loss + d_loss
                loss.backward()
                optimizer.step()

                mean_loss.append(self.to_data(loss))
                mean_tr_d_loss.append(self.to_data(d_loss))
                mean_tr_ae_loss.append(self.to_data(ae_loss))

                if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_count - 1):
                    # Net.eval()
                    losses.append(np.mean(mean_loss))
                    utils.show_progressbar([batch_idx, batch_count], mean_loss=np.mean(mean_loss))

                    pre_labels = utils.predict(lambda x: Net(x)[-1].view([x.shape[0], -1]).mm(W).view([x.shape[0], -1]), self.valid_data[view_id], self.batch_size).reshape([self.valid_data[view_id].shape[0], -1])
                    valid_labels = self.to_one_hot(self.valid_labels[view_id])
                    valid_d_loss = self.to_data(criterion(self.to_var(torch.tensor(pre_labels)), valid_labels))
                    if valid_loss_min > valid_d_loss and not self.just_valid:
                        valid_loss_min = valid_d_loss
                        valid_pre = utils.predict(lambda x: Net(x)[-1].view([x.shape[0], -1]), self.valid_data[view_id], self.batch_size).reshape([self.valid_data[view_id].shape[0], -1])
                        test_pre = utils.predict(lambda x: Net(x)[-1].view([x.shape[0], -1]), self.test_data[view_id], self.batch_size).reshape([self.test_data[view_id].shape[0], -1])
                    elif self.just_valid:
                        tr_d_loss.append(np.mean(mean_tr_d_loss))
                        val_d_loss.append(valid_d_loss)
                        tr_ae_loss.append(np.mean(mean_tr_ae_loss))
                elif batch_idx == batch_count - 1:
                    utils.show_progressbar([batch_idx, batch_count], mean_loss=np.mean(mean_loss))
                    losses.append(np.mean(mean_loss))
                else:
                    utils.show_progressbar([batch_idx, batch_count], loss=loss)
                k += 1
        if not self.just_valid:
            # self.resutls[view_id] = [valid_pre, test_pre]
            sio.savemat('features/' + self.datasets + '_' + str(view_id) + '.mat', {'valid_fea': valid_pre, 'valid_lab': self.valid_labels[view_id], 'test_fea': test_pre, 'test_lab': self.test_labels[view_id]})
            return [valid_pre, test_pre]
        else:
            self.tr_d_loss[view_id] = tr_d_loss
            self.tr_ae_loss[view_id] = tr_ae_loss
            self.val_d_loss[view_id] = val_d_loss
