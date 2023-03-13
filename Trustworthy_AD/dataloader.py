from __future__ import print_function, division
import numpy as np
from scipy.interpolate import interp1d

import Trustworthy_AD.misc as misc


def typecheck(interp_func):
    def func_wrapper(month_true, val_true, val_default, month_interp):
        assert isinstance(month_true, np.ndarray)
        assert isinstance(val_true, np.ndarray)
        assert isinstance(month_interp, np.ndarray)
        assert month_true.dtype == month_interp.dtype == int
        assert month_true.shape == val_true.shape
        assert np.all(month_true[1:] > month_true[:-1]), 'not sorted?'
        assert np.all(month_interp[1:] > month_interp[:-1]), 'not sorted?'
        return interp_func(month_true, val_true, val_default, month_interp)

    return func_wrapper


def mask_and_reference(interp_func):
    def func_wrapper(month_true, val_true, val_default, month_interp):
        has_data = ~np.isnan(val_true)

        mask = np.in1d(month_interp, month_true[has_data].astype(np.int))
        truth = np.full(month_interp.shape, np.nan)
        truth[mask] = val_true[has_data]

        interp = interp_func(month_true, val_true, val_default, month_interp)

        return interp, mask, truth

    return func_wrapper


@typecheck
@mask_and_reference
def bl_fill(month_true, val_true, val_default, month_interp):
    """ Model filling
    fill only first timepoints, other timepoints are filled by the model
    """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    # fill first timepoint
    interp[0] = valid_y[0] if sum(has_data) else val_default

    # find timepoints in both valid_x and month_interp
    common_tps = month_interp[np.in1d(month_interp, valid_x)]
    interp[np.in1d(month_interp, common_tps)] = \
        valid_y[np.in1d(valid_x, common_tps)]

    return interp


@typecheck
@mask_and_reference
def ff_fill(month_true, val_true, val_default, month_interp):
    """ Forward filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    if len(valid_y) == 0:
        interp[:] = val_default
    elif len(valid_y) == 1:
        interp[:] = valid_y[0]
    else:
        interp_fn = interp1d(
            valid_x, valid_y, kind='previous', fill_value='extrapolate')
        interp[:] = interp_fn(month_interp)

    return interp


@typecheck
@mask_and_reference
def neighbor_fill(month_true, val_true, val_default, month_interp):
    """ Nearest-neighbor filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    if len(valid_y) == 0:
        interp[:] = val_default
    elif len(valid_y) == 1:
        interp[:] = valid_y[0]
    else:
        interp_fn = interp1d(
            valid_x, valid_y, kind='nearest', fill_value='extrapolate')
        interp[:] = interp_fn(month_interp)

    return interp


def valid(time_array, tmax, tmin):
    return time_array[(time_array >= tmin) & (time_array <= tmax)]


@typecheck
@mask_and_reference
def ln_fill_partial(month_true, val_true, val_default, month_interp):
    """ Mixed model-linear filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    interp[0] = valid_y[0] if sum(has_data) else val_default

    if len(valid_y) == 1:
        # will be different from previous line when valid_x is not the first tp
        interp[np.in1d(month_interp, valid_x[0])] = valid_y[0]
    elif len(valid_y) > 1:
        interp_fn = interp1d(valid_x, valid_y, kind='linear')
        timepoints = valid(month_true, valid_x[-1], valid_x[0]).astype(np.int)
        mask = np.in1d(month_interp, timepoints)
        interp[mask] = interp_fn(month_interp[mask])

    return interp


@typecheck
@mask_and_reference
def ln_fill_full(month_true, val_true, val_default, month_interp):
    """ Linear filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    interp[0] = valid_y[0] if sum(has_data) else val_default

    if len(valid_y) == 1:
        # will be different from previous line when valid_x is not the first tp
        interp[np.in1d(month_interp, valid_x[0])] = valid_y[0]
    elif len(valid_y) > 1:
        interp_fn = interp1d(valid_x, valid_y, kind='linear')
        timepoints = valid(month_interp, valid_x[-1],
                           valid_x[0]).astype(np.int)
        interp[np.in1d(month_interp, timepoints)] = interp_fn(timepoints)

    return interp


STRATEGIES = {
    'forward': ff_fill,
    'neighbor': neighbor_fill,
    'mixed': ln_fill_partial,
    'model': bl_fill
}


def extract(frame, strategy, features, defaults):
    interp_fn = STRATEGIES[strategy]

    fields = ['Month_bl', 'DX'] + features
    ret = dict()
    for rid, sframe in misc.get_data_dict(frame, features).items():
        xin = sframe.index.values.astype(np.int)

        assert len(xin) == len(set(xin)), rid
        xin -= xin[0]
        xout = np.arange(xin[-1] - xin[0] + 1)

        in_seqs = {'Month_bl': xout}
        mk_seqs = {'Month_bl': np.zeros(len(xout), dtype=bool)}
        th_seqs = {'Month_bl': np.full(len(xout), np.nan)}

        for f in fields[1:]:
            yin = sframe[f].values
            in_seqs[f], mk_seqs[f], th_seqs[f] = interp_fn(xin, yin, defaults[f], xout)

        ret[rid] = {'input': np.array([in_seqs[f] for f in fields]).T}
        ret[rid]['mask'] = np.array([mk_seqs[f] for f in fields]).T
        ret[rid]['truth'] = np.array([th_seqs[f] for f in fields]).T

        assert ret[rid]['input'].shape == ret[rid]['mask'].shape == ret[rid]['truth'].shape

    return ret, fields


class Sorted(object):
    """
    An dataloader class for test/evaluation
    The subjects are sorted in ascending order according to subject IDs.
    """

    def __init__(self, data, batch_size, attributes):
        self.data = data[0]
        self.fields = np.array(data[1])
        self.batch_size = batch_size
        self.attributes = attributes
        self.subjects = np.sort(np.array(list(self.data.keys())))  #RID~~~~~~~~~~~
        self.idx = 0

        self.mask = {}
        self.mask['tp'] = self.fields == 'Month_bl'
        self.mask['cat'] = self.fields == 'DX'
        self.mask['val'] = np.zeros(shape=self.fields.shape, dtype=bool)
        for field in self.attributes:
            self.mask['val'] |= self.fields == field

        assert not np.any(self.mask['tp'] & self.mask['val']), 'overlap'
        assert not np.any(self.mask['cat'] & self.mask['val']), 'overlap'

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.subjects) / self.batch_size))

    # must give a deep copy of the training data !important
    def __next__(self):
        if self.idx == len(self.subjects):
            self.idx = 0
            raise StopIteration()

        rid = self.subjects[self.idx]
        self.idx += 1

        subj_data = {'rid': rid}
        seq = self.data[rid]['input']
        for k, mask in self.mask.items():
            subj_data[k] = seq[:, mask]

        return subj_data

    def value_fields(self):
        return self.fields[self.mask['val']]


def batch(matrices):
    """
    matrices 一个batch的RID对应的数据（掩码/真实值） 即一个张量  batch个RID*时间*特征
    张量中每个矩阵的行数不一样 要把他们补齐
    """
    maxlen = max(len(m) for m in matrices)      # 张量中矩阵的行数的最大值
    ret = [np.pad(m, [(0, maxlen - len(m)), (0, 0)], 'constant')[:, None, :] for m in matrices]
    # 对张量中每个矩阵进行补齐 使得每个矩阵的行数都是最大值
    # 形状：最长时间*batch*特征
    return np.concatenate(ret, axis=1)      # 按照第二个维度concat起来


class Random(Sorted):
    """
    An dataloader class for training
    The subjects are shuffled randomly in every epoch.
    """

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)
        self.rng = np.random.RandomState(seed=0)


    def __next__(self):
        # 迭代器   训练时使用数据生成batch会调用这个函数
        if self.idx == len(self.subjects):
            self.rng.shuffle(self.subjects)
            self.idx = 0        # 当前的RID索引
            raise StopIteration()

        rid_list = self.subjects[self.idx:self.idx + self.batch_size]  # 一个batch_size大小的RID集合
        self.idx += len(rid_list)       # 索引增加一个batch

        input_batch = batch([self.data[rid]['input'] for rid in rid_list])  # 数据信息也是batch_size大小
        mask_batch = batch([self.data[rid]['mask'] for rid in rid_list])
        truth_batch = batch([self.data[rid]['truth'] for rid in rid_list])

        subj_data = {}
        for k, mask in self.mask.items():  # 对mask中的3项遍历  分别是 时间 诊断分类 特征数据
            subj_data[k] = input_batch[:, :, mask]  # input_batch 第一维：时间 第二维：batch 第三维：特征
        subj_data['cat_msk'] = mask_batch[:, :, self.mask['cat']]       # 标记哪些是没有数据的 不计算loss
        subj_data['val_msk'] = mask_batch[:, :, self.mask['val']]
        subj_data['true_cat'] = truth_batch[:, :, self.mask['cat']]     # 有数据的真实值
        subj_data['true_val'] = truth_batch[:, :, self.mask['val']]

        return subj_data
