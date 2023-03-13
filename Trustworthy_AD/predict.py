from __future__ import print_function
import argparse
import json
import os
import pickle

import numpy as np
import torch

import Trustworthy_AD.misc as misc
from Trustworthy_AD.model import MODEL_DICT


def predict_subject(model, cat_seq, value_seq, time_seq):
    in_val = np.full((len(time_seq), ) + value_seq.shape[1:], np.nan)
    in_val[:len(value_seq)] = value_seq

    in_cat = np.full((len(time_seq), ) + cat_seq.shape[1:], np.nan)
    in_cat[:len(cat_seq)] = cat_seq

    with torch.no_grad():
        out_cat, out_val = model(in_cat, in_val)
    out_cat = out_cat.cpu().numpy()
    out_val = out_val.cpu().numpy()

    assert out_cat.shape[1] == out_val.shape[1] == 1

    return out_cat, out_val


def predict(model, dataset, pred_start, duration, baseline):
    model.eval()

    ret = {'subjects': dataset.subjects}
    ret['DX'] = []  # 1. likelihood of NL, MCI, and Dementia
    ret['ADAS13'] = []  # 2. (best guess, upper and lower bounds on 50% CI)
    ret['Ventricles'] = []  # 3. (best guess, upper and lower bounds on 50% CI)
    ret['dates'] = misc.make_date_col(
        [pred_start[s] for s in dataset.subjects], duration)

    col = ['ADAS13', 'Ventricles', 'ICV']
    indices = misc.get_index(list(dataset.value_fields()), col)
    mean = model.mean[col].values.reshape(1, -1)
    stds = model.stds[col].values.reshape(1, -1)

    for data in dataset:
        rid = data['rid']
        all_tp = data['tp'].squeeze(axis=1)
        start = misc.month_between(pred_start[rid], baseline[rid])
        assert np.all(all_tp == np.arange(len(all_tp)))
        mask = all_tp < start
        itime = np.arange(start + duration)
        icat = np.asarray(
            [misc.to_categorical(c, 3) for c in data['cat'][mask]])
        ival = data['val'][:, None, :][mask]

        ocat, oval = predict_subject(model, icat, ival, itime)   # EDL输出的是 alpha序列
        oval = oval[-duration:, 0, indices] * stds + mean

        ret['DX'].append(ocat[-duration:, 0, :])
        ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret['Ventricles'].append(
            misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))

    return ret


def load_checkpoint(folder):
    config = os.path.join(folder, 'config.json')
    with open(config) as fhandler:
        config = json.load(fhandler)

    feat_stats = os.path.join(folder, 'feat_stats.pkl')
    with open(feat_stats) as fhandler:
        mean, stds = pickle.load(fhandler)

    model = MODEL_DICT[config['model']](
        nb_classes=3,
        nb_measures=len(mean),
        nb_layers=config['nb_layers'],
        h_size=config['h_size'],
        h_drop=config['h_drop'],
        i_drop=config['i_drop'])

    model_weights = os.path.join(folder, 'model_weights.pt')
    model.load_state_dict(torch.load(model_weights))

    setattr(model, 'mean', mean)
    setattr(model, 'stds', stds)

    return model


def get_arg_parser(i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c',
                        default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/Trustworthy_AD/Trustworthy_AD'
                                '/output/Ours/model' + str(
                            i) + '.pt')  # 模型
    parser.add_argument('--data',
                        default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                'Trustworthy_AD/Trustworthy_AD/testdata/test' + str(
                            i) + '.pkl')  # 数据 训练数据中也存放了验证集
    parser.add_argument('--prediction', '-p',
                        default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                'Trustworthy_AD/Trustworthy_AD/output/Ours/prediction-kl-测试' + str(
                            i) + '.csv')  # 预测的保存
    parser.add_argument('--EDL', action='store_true', default=True)  # 是否使用不确定性预测

    return parser


def main():
    for i in range(20):
        args = get_arg_parser(i).parse_args()

        print('1>> ', args.data)
        print('2>> ', args.checkpoint)
        print('3>> ', args.prediction)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if args.checkpoint.endswith('.pt'):
            model = torch.load(args.checkpoint)  # 打开模型
        else:
            model = load_checkpoint(args.checkpoint)  # 不能用pytorch打开
        model.to(device)

        with open(args.data, 'rb') as fhandler:
            data = pickle.load(fhandler)  # 打开数据

        prediction = predict(model, data['test'], data['pred_start'], data['duration'], data['baseline'])
        misc.build_pred_frame(prediction, args.EDL, args.prediction)


if __name__ == '__main__':
    main()