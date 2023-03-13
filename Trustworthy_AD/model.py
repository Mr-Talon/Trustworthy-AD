import numpy as np
import torch
import torch.nn as nn

from Trustworthy_AD.rnn import MinimalRNNCell
from Trustworthy_AD.rnn import LssCell
from Trustworthy_AD.rnn import FastGRNNCell
from Trustworthy_AD.rnn import GRUCell
from Trustworthy_AD.rnn import Atte_FastGRNNCell
from Trustworthy_AD.rnn import Atte_MinimalRNNCell
from Trustworthy_AD.rnn import Atte_GRUCell


def jozefowicz_init(forget_gate):
    forget_gate.data.fill_(1)


class RnnModelInterp(torch.nn.Module):
    def __init__(self, celltype, nb_classes, nb_measures, h_size, MLP, MLP_hid, EDL, attention, dropout, **kwargs):
        super(RnnModelInterp, self).__init__()
        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']
        self.MLP = MLP
        self.EDL = EDL

        if MLP:  # MLP
            self.hid2MlP = nn.Linear(h_size, MLP_hid)
            self.hid2category = nn.Linear(MLP_hid, nb_classes)
            self.hid2measures = nn.Linear(MLP_hid, nb_measures)
        else:
            self.hid2category = nn.Linear(h_size, nb_classes)
            self.hid2measures = nn.Linear(h_size, nb_measures)

        self.cells = nn.ModuleList()
        if attention:       # 使用Attention的RNNCell 里面还有Attention结构 需要额外传入dropout值
            self.cells.append(celltype(nb_classes + nb_measures, h_size, dropout))
            for _ in range(1, kwargs['nb_layers']):
                self.cells.append(celltype(h_size, h_size, dropout))
        else:
            self.cells.append(celltype(nb_classes + nb_measures, h_size))
            for _ in range(1, kwargs['nb_layers']):
                self.cells.append(celltype(h_size, h_size))

    def init_hidden_state(self, batch_size):
        raise NotImplementedError

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device

        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)

        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    def forward(self, _cat_seq, _val_seq):
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(_val_seq.shape[1])
        masks = self.dropout_mask(_val_seq.shape[1])

        cat_seq = _cat_seq.copy()
        val_seq = _val_seq.copy()

        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], masks, hidden)
            # 使用EDL o_cat输出的是Dirichlet分布的参数
            # 不使用EDL o_cat输出的是类别分布的参数 即softmax之后的结果
            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = np.isnan(val_seq[j])
            val_seq[j][idx] = o_val.data.cpu().numpy()[idx]

            if self.EDL:
                idx = np.isnan(cat_seq[j])
                strength = torch.sum(o_cat.data, dim=1, keepdim=True)  # 输出是Dirichlet分布的参数alpha 对其求和得到狄利克雷强度
                prob = o_cat.data / strength  # 类别分布的参数是Dirichlet分布的均值
                cat_seq[j][idx] = prob.cpu().numpy()[idx]  # 模型补齐依旧是用输出的类别分布补齐 【为什么不是转换成one-hot？？？？？？？？？？？？？？】
            else:
                idx = np.isnan(cat_seq[j])
                cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

        return torch.stack(out_cat_seq), torch.stack(out_val_seq)


class SingleStateRNN(RnnModelInterp):
    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state  # 最终返回一个列表

    def predict(self, i_cat, i_val, masks, hid):
        i_mask, r_mask = masks
        h_t = torch.cat([hid[0].new(i_cat), hid[0].new(i_val) * i_mask], dim=-1)

        next_hid = []
        for cell, prev_h, mask in zip(self.cells, hid, r_mask):
            h_t = cell(h_t, prev_h * mask)
            next_hid.append(h_t)

        if self.MLP:  # !!!! MLP
            h_t = torch.relu(self.hid2MlP(h_t))

        if self.EDL:
            o_cat = torch.relu(self.hid2category(h_t)) + 1  # Dirichlet 输出经过relu后得到evidence +1 得到alpha
            o_val = self.hid2measures(h_t) + hid[0].new(i_val)
        else:
            o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
            o_val = self.hid2measures(h_t) + hid[0].new(i_val)

        return o_cat, o_val, next_hid


class Atte_SingleStateRNN(RnnModelInterp):
    def init_hidden_state(self, batch_size):
        # 为每层RNN 初始化context【输入到RNN的是context和】 和 hid列表【作为Attention的键值对】
        dev = next(self.parameters()).device
        context = []
        hid_list = []
        for cell in self.cells:
            context.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
            hid_list.append([])
        return context, hid_list

    def forward(self, _cat_seq, _val_seq):
        out_cat_seq, out_val_seq = [], []

        context, hid_list = self.init_hidden_state(_val_seq.shape[1])
        masks = self.dropout_mask(_val_seq.shape[1])

        cat_seq = _cat_seq.copy()
        val_seq = _val_seq.copy()

        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, context = self.predict(cat_seq[i], val_seq[i], masks, context, hid_list)  # 使用Attention的RNN 第一层 输入是input和context，以及hid之前所有时间的hid
            # 使用EDL o_cat输出的是Dirichlet分布的参数
            # 不使用EDL o_cat输出的是类别分布的参数 即softmax之后的结果

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # 模型补齐 回归直接补
            idx = np.isnan(val_seq[j])
            val_seq[j][idx] = o_val.data.cpu().numpy()[idx]

            # 分类EDL要求均值
            if self.EDL:
                idx = np.isnan(cat_seq[j])
                strength = torch.sum(o_cat.data, dim=1, keepdim=True)  # 输出是Dirichlet分布的参数alpha 对其求和得到狄利克雷强度
                prob = o_cat.data / strength  # 类别分布的参数是Dirichlet分布的均值
                cat_seq[j][idx] = prob.cpu().numpy()[idx]  # 模型补齐依旧是用输出的类别分布补齐 【为什么不是转换成one-hot？？？？？？？？？？？？？？】
            else:
                idx = np.isnan(cat_seq[j])
                cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

        return torch.stack(out_cat_seq), torch.stack(out_val_seq)

    def predict(self, i_cat, i_val, masks, context, hid_list):
        # 对一个时间的数据过模型

        i_mask, r_mask = masks
        # 输入初始化
        h_t = torch.cat([context[0].new(i_cat), context[0].new(i_val) * i_mask], dim=-1)  # RNN的input 第一层是x 后面就是各层RNN的hid

        next_context = []   # context会被下一个时间使用 hid会被存在hid_list里
        # 过RNNCell
        for cell, mask, prev_context, hid_layer_list in zip(self.cells, r_mask, context, hid_list):
            # 遍历RNN层、当前层的输入、掩码、上下文、当前层的hid_list

            h_t, context_t = cell(h_t, prev_context * mask, hid_layer_list)    # 输出 下一层RNN的输入 下一个时间的一层的context
            next_context.append(context_t)

        # 过MLP
        if self.MLP:  # !!!! MLP
            h_t = torch.relu(self.hid2MlP(h_t))

        # 过EDL/最后一层输出
        if self.EDL:
            o_cat = torch.relu(self.hid2category(h_t)) + 1  # Dirichlet 输出经过relu后得到evidence +1 得到alpha
            o_val = self.hid2measures(h_t) + context[0].new(i_val)
        else:
            o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
            o_val = self.hid2measures(h_t) + context[0].new(i_val)

        return o_cat, o_val, next_context


class FastGRNN(SingleStateRNN):
    """ FastGRNN """

    def __init__(self, **kwargs):
        super(FastGRNN, self).__init__(FastGRNNCell, **kwargs)


class GRU(SingleStateRNN):
    """ GRU """

    def __init__(self, **kwargs):
        super(GRU, self).__init__(GRUCell, **kwargs)


class MinimalRNN(SingleStateRNN):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super(MinimalRNN, self).__init__(MinimalRNNCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(cell.bias_hh)


class LSS(SingleStateRNN):
    ''' Linear State-Space '''

    def __init__(self, **kwargs):
        super(LSS, self).__init__(LssCell, **kwargs)


class LSTM(RnnModelInterp):
    ''' LSTM '''

    def __init__(self, **kwargs):
        super(LSTM, self).__init__(nn.LSTMCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(
                cell.bias_hh[cell.hidden_size:cell.hidden_size * 2])

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            h_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            c_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            state.append((h_x, c_x))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = torch.cat([hid[0][0].new(i_cat), hid[0][0].new(i_val) * i_mask],
                        dim=-1)

        states = []
        for cell, prev_state, mask in zip(self.cells, hid, r_mask):
            h_t, c_t = cell(h_t, (prev_state[0] * mask, prev_state[1]))
            states.append((h_t, c_t))

        if self.MLP:  # !!!! MLP
            h_t = torch.relu(self.hid2MlP(h_t))

        if self.EDL:
            o_cat = torch.relu(self.hid2category(h_t)) + 1  # Dirichlet 输出经过relu后得到evidence +1 得到alpha
            o_val = self.hid2measures(h_t) + hid[0].new(i_val)
        else:
            o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
            o_val = self.hid2measures(h_t) + hid[0].new(i_val)

        return o_cat, o_val, states


class Atte_FastGRNN(Atte_SingleStateRNN):
    """ attention FastGRNN """

    def __init__(self, **kwargs):
        super(Atte_FastGRNN, self).__init__(Atte_FastGRNNCell, **kwargs)


class Atte_MinimalRNN(Atte_SingleStateRNN):
    """ attention FastGRNN """

    def __init__(self, **kwargs):
        super(Atte_MinimalRNN, self).__init__(Atte_MinimalRNNCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(cell.bias_hh)


class Atte_GRU(Atte_SingleStateRNN):
    """ attention FastGRNN """

    def __init__(self, **kwargs):
        super(Atte_GRU, self).__init__(Atte_GRUCell, **kwargs)


MODEL_DICT = {'LSTM': LSTM, 'MinRNN': MinimalRNN, 'LSS': LSS, 'FastGRNN': FastGRNN, 'GRU': GRU,
              'Atte_FastGRNN':Atte_FastGRNN, 'Atte_MinimalRNN':Atte_MinimalRNN, 'Atte_GRU':Atte_GRU}
