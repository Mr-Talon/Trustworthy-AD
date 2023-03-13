import math
import torch
import torch.nn as nn

from Trustworthy_AD.attention import AdditiveAttention


class MinimalRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinimalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)

        self.weight_uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_uh.data.uniform_(-stdv, stdv)
        self.weight_uz.data.uniform_(-stdv, stdv)

        self.bias_hh.data.uniform_(stdv)

    def forward(self, input, hx):
        z = torch.tanh(self.W(input))
        u = torch.addmm(self.bias_hh, hx, self.weight_uh)
        u = torch.addmm(u, z, self.weight_uz)
        u = torch.sigmoid(u)
        return u * hx + (1 - u) * z


class LssCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LssCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_i2h.data.uniform_(-stdv, stdv)
        self.weight_h2h.data.uniform_(-stdv, stdv)

        self.bias.data.uniform_(stdv)

    def forward(self, input, hx):
        h = torch.addmm(self.bias, input, self.weight_i2h)
        h = torch.addmm(h, hx, self.weight_h2h)
        return h


class FastGRNNCell(nn.Module):
    '''
    z_t = sigmoid(Wx_t + Uh_{t-1} + B_g)
    h_t^ = tanh(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^
    '''

    def __init__(self, input_size, hidden_size, zetaInit=1.0, nuInit=-4.0):
        super(FastGRNNCell, self).__init__()
        self._zetaInit = zetaInit
        self._nuInit = nuInit

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1]))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1]))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)


    def forward(self, input, state):
        wComp = torch.matmul(input, self.W)
        uComp = torch.matmul(state, self.U)

        z = torch.sigmoid(wComp + uComp + self.bias_gate)
        c = torch.tanh(wComp + uComp + self.bias_update)
        new_h = z * state + \
                (torch.sigmoid(self.zeta) * (1.0 - z) + torch.sigmoid(self.nu)) * c

        return new_h


class GRUCell(nn.Module):
    '''
    r_t = gate_nl(W1x_t + U1h_{t-1} + B_r)
    z_t = gate_nl(W2x_t + U2h_{t-1} + B_g)
    h_t^ = update_nl(W3x_t + r_t*U3(h_{t-1}) + B_h)
    h_t = z_t*h_{t-1} + (1-z_t)*h_t^
    '''

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W2 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W3 = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.U1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U3 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_r = nn.Parameter(torch.zeros([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.zeros([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.zeros([1, hidden_size]))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W1.data.uniform_(-stdv, stdv)
        self.W2.data.uniform_(-stdv, stdv)
        self.W3.data.uniform_(-stdv, stdv)

        self.U1.data.uniform_(-stdv, stdv)
        self.U2.data.uniform_(-stdv, stdv)
        self.U3.data.uniform_(-stdv, stdv)


    def forward(self, input, state):
        wComp1 = torch.matmul(input, self.W1)
        wComp2 = torch.matmul(input, self.W2)
        wComp3 = torch.matmul(input, self.W3)

        uComp1 = torch.matmul(state, self.U1)
        uComp2 = torch.matmul(state, self.U2)

        r = torch.sigmoid(wComp1 + uComp1 + self.bias_r)
        z = torch.sigmoid(wComp2 + uComp2 + self.bias_gate)

        c = torch.tanh(wComp3 + torch.matmul(r * state, self.U3) + self.bias_update)

        new_h = z * state + (1.0 - z) * c
        return new_h


class Atte_FastGRNNCell(nn.Module):
    '''
    z_t = sigmoid(Wx_t + Uh_{t-1} + B_g)
    h_t^ = tanh(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^
    '''

    def __init__(self, input_size, hidden_size, dropout, zetaInit=1.0, nuInit=-4.0):
        super(Atte_FastGRNNCell, self).__init__()
        self._zetaInit = zetaInit
        self._nuInit = nuInit

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

        self.zeta = nn.Parameter(self._zetaInit * torch.ones([1, 1]))
        self.nu = nn.Parameter(self._nuInit * torch.ones([1, 1]))

        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout)  # RNNCell 增加一个加性Attention

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)


    def forward(self, input, context, hid_list):
        wComp = torch.matmul(input, self.W)
        uComp = torch.matmul(context, self.U)  # 使用Attention hid是关于输入input和上下文context的函数

        z = torch.sigmoid(wComp + uComp + self.bias_gate)
        c = torch.tanh(wComp + uComp + self.bias_update)
        new_h = z * context + \
                (torch.sigmoid(self.zeta) * (1.0 - z) + torch.sigmoid(self.nu)) * c

        hid_list.append(new_h)
        hid_list = torch.stack(hid_list)

        # 用注意力机制生成当前时间的context
        query = torch.unsqueeze(new_h, dim=1)
        hid_list = hid_list.permute(1, 0, 2)
        context_t = self.attention(query, hid_list, hid_list)
        context_t = context_t.squeeze(1)

        return new_h, context_t


class Atte_MinimalRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super(Atte_MinimalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)

        self.weight_uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout)  # RNNCell 增加一个加性Attention

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_uh.data.uniform_(-stdv, stdv)
        self.weight_uz.data.uniform_(-stdv, stdv)

        self.bias_hh.data.uniform_(stdv)

    def forward(self, input, context, hid_list):
        z = torch.tanh(self.W(input))
        u = torch.addmm(self.bias_hh, context, self.weight_uh)
        u = torch.addmm(u, z, self.weight_uz)
        u = torch.sigmoid(u)
        new_h = u * context + (1 - u) * z

        hid_list.append(new_h)
        hid_list = torch.stack(hid_list)

        # 用注意力机制生成当前时间的context
        query = torch.unsqueeze(new_h, dim=1)
        hid_list = hid_list.permute(1, 0, 2)
        context_t = self.attention(query, hid_list, hid_list)
        context_t = context_t.squeeze(1)

        return new_h, context_t


class Atte_GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super(Atte_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W2 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W3 = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.U1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U3 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_r = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_gate = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))

        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout)  # RNNCell 增加一个加性Attention

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W1.data.uniform_(-stdv, stdv)
        self.W2.data.uniform_(-stdv, stdv)
        self.W3.data.uniform_(-stdv, stdv)

        self.U1.data.uniform_(-stdv, stdv)
        self.U2.data.uniform_(-stdv, stdv)
        self.U3.data.uniform_(-stdv, stdv)


    def forward(self, input, context, hid_list):
        wComp1 = torch.matmul(input, self.W1)
        wComp2 = torch.matmul(input, self.W2)
        wComp3 = torch.matmul(input, self.W3)

        uComp1 = torch.matmul(context, self.U1)
        uComp2 = torch.matmul(context, self.U2)

        r = torch.sigmoid(wComp1 + uComp1 + self.bias_r)
        z = torch.sigmoid(wComp2 + uComp2 + self.bias_gate)

        c = torch.tanh(wComp3 + torch.matmul(r * context, self.U3) + self.bias_update)

        new_h = z * context + (1.0 - z) * c

        hid_list.append(new_h)
        hid_list = torch.stack(hid_list)

        # 用注意力机制生成当前时间的context
        query = torch.unsqueeze(new_h, dim=1)
        hid_list = hid_list.permute(1, 0, 2)
        context_t = self.attention(query, hid_list, hid_list)
        context_t = context_t.squeeze(1)

        return new_h, context_t
