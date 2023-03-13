from __future__ import print_function, division
import argparse
import json
import time
import pickle
import copy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import Trustworthy_AD.misc as misc
from Trustworthy_AD.model import MODEL_DICT
from loss import evidential_classification
from predict import predict
from misc import build_pred_frame
from evaluation import eval_submission


def ent_loss(pred, true, mask):
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]

    return torch.nn.functional.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects


def edl_loss(alpha, true, mask, epoch, total_epoch):
    """
    贝叶斯风险损失
    输入：
        pred 分输出的Dirichlet分布参数alpha  [nb_timpoints：时间, nb_subjects：batch, nb_classes]
        true 分类标签    [nb_timpoints：时间, nb_subjects：batch, 1]
        mask 掩码 是否需要计算损失 [nb_timpoints, nb_subjects, 1]
    """
    assert isinstance(alpha, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]     # batch大小

    alpha = alpha.reshape(alpha.size(0) * alpha.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = alpha.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = alpha[alpha.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]

    return evidential_classification(o_pred, o_true, epoch, total_epoch) / nb_subjects  # 返回的是所有batch 所有时间的loss和 minibatch对batch求平均


def mae_loss(pred, true, mask):

    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.bool)
    assert pred.shape == indices.shape
    pred[indices] = 0

    return torch.nn.functional.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects


def to_cat_seq(labels):

    return np.asarray([misc.to_categorical(c, 3) for c in labels])


def train_1epoch(args, model, dataset, optimizer, epoch, total_epoch):
    model.train()
    total_ent = total_mae = 0
    for iteration, batch in enumerate(dataset):
        if len(batch['tp']) == 1:
            continue

        optimizer.zero_grad()
        pred_cat, pred_val = model(to_cat_seq(batch['cat']), batch['val'])

        mask_cat = batch['cat_msk'][1:]
        assert mask_cat.sum() > 0

        if args.EDL:  # EDL损失函数不一样
            ent = edl_loss(pred_cat, batch['true_cat'][1:], mask_cat, epoch, total_epoch)
        else:
            ent = ent_loss(pred_cat, batch['true_cat'][1:], mask_cat)
        mae = mae_loss(pred_val, batch['true_val'][1:], batch['val_msk'][1:])
        total_loss = mae + args.w_ent * ent

        total_loss.backward()
        optimizer.step()
        batch_size = mask_cat.shape[1]
        total_ent += ent.item() * batch_size
        total_mae += mae.item() * batch_size

    return total_ent / len(dataset.subjects), total_mae / len(dataset.subjects)


def save_config(args, config_path):
    with open(config_path, 'w') as fhandler:
        print(json.dumps(vars(args), sort_keys=True), file=fhandler)


def train(args):
    log = print if args.verbose else lambda *x, **i: None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    write = SummaryWriter(log_dir=args.record, flush_secs=60)  # tensorboard 记录

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)
    nb_measures = len(data['train'].value_fields())

    model_class = MODEL_DICT[args.model]
    model = model_class(
        nb_classes=3,
        nb_measures=nb_measures,
        nb_layers=args.nb_layers,
        MLP=args.MLP,
        MLP_hid=args.MLP_hid,
        EDL=args.EDL,
        attention=args.attention,
        dropout=args.dropout,
        h_size=args.h_size,
        h_drop=args.h_drop,
        i_drop=args.i_drop)
    setattr(model, 'mean', data['mean'])
    setattr(model, 'stds', data['stds'])

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    log(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    mAUC_best = 1e-8
    BCA_best = 1e-8
    ECE_best = 10000
    start = time.time()
    try:
        for i in range(args.epochs):
            loss = train_1epoch(args, model, data['train'], optimizer, i + 1, args.epochs)
            write.add_scalar('loss/Train_epoch_cls', loss[0], i)
            write.add_scalar('loss/Train_epoch_reg', loss[1], i)

            log_info = (i + 1, args.epochs, misc.time_from(start)) + loss
            log('%d/%d %s \t分类ENT %.3f\t回归MAE %.3f' % log_info)

            # 验证集
            if (i % 10 == 0) | (i == args.epochs - 1):
                prediction = predict(model, data['test'], data['pred_start'], data['duration'], data['baseline'])
                build_pred_frame(prediction, args.EDL, args.temp_prediction)  # 预测输出的csv临时存放 会被覆盖
                result = eval_submission(misc.read_csv(args.reference), misc.read_csv(args.temp_prediction), args.EDL,
                                         draw=False, total_epoch=args.epochs)
                write.add_scalar('metric/val_mAUC', result['mAUC'], i)
                write.add_scalar('metric/val_BCA', result['bca'], i)
                write.add_scalar('metric/val_cls_loss', result['cls_loss'], i)
                write.add_scalar('metric/val_MAE_ADAS', result['adasMAE'], i)
                write.add_scalar('metric/val_MAE_V', result['ventsMAE'], i)
                write.add_scalar('metric/val_ECE', result['ECE'], i)

                print('\t\tmAUC %.3f\tbca %.3f\tadasMAE %.3f\tventsMAE %.4f\tval_cls_loss %.3f\tECE %.3f' % (
                    result['mAUC'], result['bca'], result['adasMAE'], result['ventsMAE'], result['cls_loss'],
                    result['ECE']))

                mAUC_rate = (result['mAUC'] - mAUC_best) / mAUC_best  # 越高越好
                BCA_rate = (result['bca'] - BCA_best) / BCA_best
                ECE_rate = (ECE_best - result['ECE']) / ECE_best  # 越低越好
                if ((result['ECE'] >= 0.1) & (mAUC_rate + BCA_rate + 0.1 * ECE_rate > 0)) | \
                        ((result['ECE'] < 0.1) & (mAUC_rate + BCA_rate > 0)):  # ECE较大也要考虑ECE的减小，ECE变化很快 ECE小于0.1更需要注重模型性能
                    best_loss = result['cls_loss']
                    mAUC_best = result['mAUC']
                    BCA_best = result['bca']
                    ECE_best = result['ECE']
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('---------模型暂存----------')

            # scheduler.step()  # 每个epoch调整学习率
    except KeyboardInterrupt:
        print('Early exit')

    print('\n训练结束，保存最佳模型：')
    print('验证集最佳loss是：', best_loss, 'mAUC：', mAUC_best, 'BCA:', BCA_best, 'ECE:', ECE_best)
    model.load_state_dict(best_model_wts)
    torch.save(model, args.checkpoint)
    save_config(args, '%s.json' % args.checkpoint)

    return best_loss, mAUC_best, BCA_best, ECE_best


def get_arg_parser(i):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default='Atte_FastGRNN')  # 使用的模型

    # 路径
    parser.add_argument('--data', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                          'Trustworthy_AD/Trustworthy_AD/'
                                          'traindata/train' + str(i) + '.pkl')  # 训练数据
    parser.add_argument('--checkpoint', '-c', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习'
                                                      '/Trustworthy_AD/Trustworthy_AD/output'
                                                      '/Ours/model-' + str(
        i) + '.pt')  # 模型保存
    parser.add_argument('--record', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习'
                                            '/Trustworthy_AD/Trustworthy_AD/'
                                            'output/Ours/record-无cos-优化' + str(
        i))  # tensorboard 保存目录
    parser.add_argument('--temp_prediction', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习'
                                                     '/Trustworthy_AD/Trustworthy_AD/output/temp.csv')  # epoch验证输出csv临时存放路径
    parser.add_argument('--reference', '-r', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                                     'Trustworthy_AD/Trustworthy_AD/folds/fold' + str(i) + '_val.csv')  # 验证集gt

    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--scheduler_cos', type=int, default=32)  # 学习率调整
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=True)

    # 选项
    parser.add_argument('--MLP', action='store_true', default=True)  # 逻辑/线性回归为False  MLP为True
    parser.add_argument('--EDL', action='store_true', default=True)  # 是否使用不确定性预测
    parser.add_argument('--attention', action='store_true', default=True)  # 是否使用attention

    # 超参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--w_ent', type=float, default=2)  # 损失函数分类回归权重
    parser.add_argument('--nb_layers', type=int, default=2)  # rnn层数
    parser.add_argument('--h_size', type=int, default=64)  # 隐状态向量长度
    parser.add_argument('--MLP_hid', type=int, default=64)  # MLP隐藏层长度
    parser.add_argument('--dropout', type=float, default=0.5)  # attention drop_out
    parser.add_argument('--i_drop', type=float, default=0.1)  # 模型输入drop_out
    parser.add_argument('--h_drop', type=float, default=0.2)  # RNNdrop_out
    parser.add_argument('--weight_decay', type=float, default=5e-07)  # l2范数权重

    return parser


def main():
    loss = []
    mAUC = []
    BCA = []
    ECE = []
    for i in range(10):
        args = get_arg_parser(i).parse_args()
        print('1>> ', args.data)
        print('2>> ', args.checkpoint)
        loss_i, mAUC_i, BCA_i, ECE_i = train(args)
        loss.append(loss_i)
        mAUC.append(mAUC_i)
        BCA.append(BCA_i)
        ECE.append(ECE_i)
    loss = np.array(loss)
    mAUC = np.array(mAUC)
    BCA = np.array(BCA)
    ECE = np.array(ECE)
    print('交叉验证结果均值：mAUC->', mAUC.mean(), 'BCA->', BCA.mean(), 'loss->', loss.mean(), 'ECE->', ECE.mean())
    print('\n')


if __name__ == '__main__':
    main()
