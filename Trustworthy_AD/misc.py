from __future__ import print_function, division
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
import numpy as np
import pandas as pd


def load_feature(feature_file_path):
    return [l.strip() for l in open(feature_file_path)]


def time_from(start):
    """ Return duration from *start* to now """
    duration = relativedelta(seconds=time.time() - start)
    return '%dm %ds' % (duration.minutes, duration.seconds)


def str2date(string):
    """ Convert string to datetime object """
    return datetime.strptime(string, '%Y-%m-%d')


def has_data_mask(frame):
    return ~frame.isnull().apply(np.all, axis=1)


def get_data_dict(frame, features):
    ret = {}
    frame_ = frame.copy()
    frame_['Month_bl'] = frame_['Month_bl'].round().astype(int)
    for subj in np.unique(frame_.RID):
        subj_data = frame_[frame_.RID == subj].sort_values('Month_bl')
        subj_data = subj_data[has_data_mask(subj_data[features])]

        subj_data = subj_data.set_index('Month_bl', drop=True)
        ret[subj] = subj_data.drop(['RID'], axis=1)
    return ret


def build_pred_frame(prediction, EDL, outpath='',):
    table = pd.DataFrame()
    dates = prediction['dates']
    table['RID'] = prediction['subjects'].repeat([len(x) for x in dates])
    table['Forecast Month'] = np.concatenate(
        [np.arange(len(x)) + 1 for x in dates])
    table['Forecast Date'] = np.concatenate(dates)

    diag = np.concatenate(prediction['DX'])
    if EDL: # 如果使用EDL 输出的是3个类别相关的Dirichlet分布参数
        table['CN relative alpha'] = diag[:, 0]
        table['MCI relative alpha'] = diag[:, 1]
        table['AD relative alpha'] = diag[:, 2]

        # 同时还要输出不确定性
        strength = np.sum(diag,axis=1)
        uncertainty = 3/strength    # 得到不确定性

        table['CN relative probability'] = diag[:, 0]/strength
        table['MCI relative probability'] = diag[:, 1]/strength
        table['AD relative probability'] = diag[:, 2]/strength
        table['uncertainty'] = uncertainty
    else:
        table['CN relative probability'] = diag[:, 0]
        table['MCI relative probability'] = diag[:, 1]
        table['AD relative probability'] = diag[:, 2]


    adas = np.concatenate(prediction['ADAS13'])
    table['ADAS13'] = adas[:, 0]
    # table['ADAS13 50% CI lower'] = adas[:, 1]
    # table['ADAS13 50% CI upper'] = adas[:, 2]

    vent = np.concatenate(prediction['Ventricles'])
    table['Ventricles_ICV'] = vent[:, 0]
    # table['Ventricles_ICV 50% CI lower'] = vent[:, 1]
    # table['Ventricles_ICV 50% CI upper'] = vent[:, 2]

    assert len(diag) == len(adas) == len(vent)

    if outpath:
        table.to_csv(outpath, index=False)

    return table


def month_between(end, start):
    """ Get duration (in months) between *end* and *start* dates """
    # assert end >= start
    diff = relativedelta(end, start)
    months = 12 * diff.years + diff.months
    to_next = relativedelta(end + relativedelta(months=1, days=-diff.days),
                            end).days
    to_prev = diff.days
    return months + (to_next < to_prev)


def make_date_col(starts, duration):
    date_range = [relativedelta(months=i) for i in range(duration)]
    ret = []
    for start in starts:
        ret.append([start + d for d in date_range])

    return ret


def get_index(fields, keys):
    """ Get indices of *keys*, each of which is in list *fields* """
    assert isinstance(keys, list)
    assert isinstance(fields, list)
    return [fields.index(k) for k in keys]


def to_categorical(y, nb_classes):
    """ Convert list of labels to one-hot vectors """
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1.

    return ret_mat


def log_result(result, path, verbose):
    """ Output result to screen/file """
    frame = pd.DataFrame([result])[['mAUC', 'bca', 'adasMAE', 'ventsMAE']]
    if verbose:
        print(frame)
    if path:
        frame.to_csv(path, index=False)


def PET_conv(value):
    '''Convert PET measures from string to float '''
    try:
        return float(value.strip().strip('>'))
    except ValueError:
        return float(np.nan)


def Diagnosis_conv(value):
    '''Convert diagnosis from string to float '''
    if value == 'CN':
        return 0.
    if value == 'MCI':
        return 1.
    if value == 'AD':
        return 2.
    return float('NaN')


def DX_conv(value):
    '''Convert change in diagnosis from string to float '''
    if isinstance(value, str):
        if value.endswith('Dementia'):
            return 2.
        if value.endswith('MCI'):
            return 1.
        if value.endswith('NL'):
            return 0.

    return float('NaN')


def add_ci_col(values, ci, lo, hi):
    """ Add lower/upper confidence interval to prediction """
    return np.clip(np.vstack([values, values - ci, values + ci]).T, lo, hi)


def censor_d1_table(_table):
    """ Remove problematic rows """
    _table.drop(3229, inplace=True)  # RID 2190, Month = 3, Month_bl = 0.45
    _table.drop(4372, inplace=True)  # RID 4579, Month = 3, Month_bl = 0.32
    _table.drop(
        8376, inplace=True)  # Duplicate row for subject 1088 at 72 months
    _table.drop(
        8586, inplace=True)  # Duplicate row for subject 1195 at 48 months
    _table.loc[
        12215,
        'Month_bl'] = 48.  # Wrong EXAMDATE and Month_bl for subject 4960
    _table.drop(10254, inplace=True)  # Abnormaly small ICV for RID 4674
    _table.drop(12245, inplace=True)  # Row without measurements, subject 5204


def load_table(csv, columns):
    """ Load CSV, only include *columns* """
    table = pd.read_csv(csv, converters=CONVERTERS, usecols=columns)
    censor_d1_table(table)

    return table


# Converters for columns with non-numeric values
CONVERTERS = {
    'CognitiveAssessmentDate': str2date,
    'ScanDate': str2date,
    'Forecast Date': str2date,
    'EXAMDATE': str2date,
    'Diagnosis': Diagnosis_conv,
    'DX': DX_conv,
    'PTAU_UPENNBIOMK9_04_19_17': PET_conv,
    'TAU_UPENNBIOMK9_04_19_17': PET_conv,
    'ABETA_UPENNBIOMK9_04_19_17': PET_conv
}


def get_baseline_prediction_start(frame):
    """ Get baseline dates and dates when prediction starts """
    one_month = relativedelta(months=1)
    baseline = {}
    start = {}
    for subject in np.unique(frame.RID):
        dates = frame.loc[frame.RID == subject, 'EXAMDATE']
        baseline[subject] = min(dates)
        start[subject] = max(dates) + one_month

    return baseline, start


def get_mask(csv_path, use_validation):
    """ Get masks from CSV file """
    columns = ['RID', 'EXAMDATE', 'train', 'val', 'test']
    frame = load_table(csv_path, columns)
    train_mask = frame.train == 1
    if use_validation:
        pred_mask = frame.val == 1
    else:
        pred_mask = frame.test == 1

    return train_mask, pred_mask, frame[pred_mask]


def read_csv(fpath):
    """ Load CSV with converters """
    return pd.read_csv(fpath, converters=CONVERTERS)
