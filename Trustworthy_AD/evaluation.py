from __future__ import print_function, division
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

import Trustworthy_AD.misc as misc
from loss import evidential_classification

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimSun', style='white', )


def a_value(probabilities, zero_label=0, one_label=1):
    """
    计算AUC值
    Args:
        probabilities 所有测试集样本  分类gt 和 各个类别的预测概率
        zero_label 第一个类别  0
        one_label  第二个类别  1
    """
    expanded_points = []
    for instance in probabilities:  # 遍历所有测试集样本   分类gt 和 各个类别的预测概率
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0], instance[1][zero_label]))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(data, no_classes):
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(no_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as
    # this gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (
                             a_value(data, zero_label=pairing[0], one_label=pairing[1]) +
                             a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(no_classes * (no_classes - 1)))  # Eqn 7


def calcBCA(estimLabels, trueLabels, no_classes):
    bcaAll = []
    for c0 in range(no_classes):
        # c0 can be either CTL, MCI or AD

        # one example when c0=CTL
        # TP - label was estimated as CTL, and the true label was also CTL
        # FP - label was estimated as CTL, but the true label was not CTL
        TP = np.sum((estimLabels == c0) & (trueLabels == c0))
        TN = np.sum((estimLabels != c0) & (trueLabels != c0))
        FP = np.sum((estimLabels == c0) & (trueLabels != c0))
        FN = np.sum((estimLabels != c0) & (trueLabels == c0))

        # sometimes the sensitivity of specificity can be NaN, if the user
        # doesn't forecast one of the classes.
        # In this case we assume a default value for sensitivity/specificity
        if (TP + FN) == 0:
            sensitivity = 0.5
        else:
            sensitivity = (1. * TP) / (TP + FN)

        if (TN + FP) == 0:
            specificity = 0.5
        else:
            specificity = (1. * TN) / (TN + FP)

        bcaCurr = 0.5 * (sensitivity + specificity)
        bcaAll += [bcaCurr]

    return np.mean(bcaAll)


def nearest(series, target):
    """ Return index in *series* with value closest to *target* """
    return (series - target).abs().idxmin()


def ece_binary(probabilities, target, n_bins=10, draw=False):
    pos_frac, mean_confidence, bin_count, non_zero_bins = _binary_calibration(target.flatten(), probabilities.flatten(),
                                                                              n_bins)
    # 返回每个bins accuracy、平均置信度、样本个数、以及本bins是否有样本

    bin_proportions = bin_count / bin_count.sum()  # ECE公式的权重
    ece = (np.abs(mean_confidence - pos_frac) * bin_proportions).sum()

    _, _, bin_count_accconf, non_zero_bins_accconf = _binary_calibration(target.flatten(), probabilities.flatten(),
                                                                         20)
    acc_total = target.sum() / len(target)  # 总精度
    confidence_total = probabilities.mean()

    if draw:
        j = 0
        ave_acc = []
        gap_bottom = []
        gap_height = []
        bin_count_total = []

        # 计算展示的数据
        x = np.arange(1 / (2 * n_bins), 1 + 1 / (2 * n_bins), 1 / (n_bins))
        # 可靠图数据
        for i in range(n_bins):
            if non_zero_bins[i]:
                ave_acc.append(pos_frac[j])
                gap_height.append(abs(pos_frac[j] - x[i]))
                if pos_frac[j] > x[i]:
                    gap_bottom.append(x[i])
                else:
                    gap_bottom.append(pos_frac[j])
                j = j + 1
            else:
                ave_acc.append(0.)
                gap_bottom.append(0.)
                gap_height.append(0.)
        # acc confi数据
        j = 0
        for i in range(20):
            if non_zero_bins_accconf[i]:
                bin_count_total.append(bin_count_accconf[j])
                j = j + 1
            else:
                bin_count_total.append(0.)

        ave_acc = np.array(ave_acc)
        gap_bottom = np.array(gap_bottom)
        gap_height = np.array(gap_height)
        bin_count_total = np.array(bin_count_total) / len(target)

        # 绘制可靠图
        # fig1 = plt.figure(figsize=(7, 6))
        # ax1 = fig1.add_subplot(111)
        fig, ax1 = plt.subplots(figsize=(11, 11))
        ax1.set_title('MinimalRNN可靠图', fontproperties='SimSun', fontsize=40)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.bar(x, ave_acc, color='#6DD5FA', width=1 / (n_bins), edgecolor='#134857', linewidth=1, label=u'Outputs')
        ax1.set_ylabel(u'Accuracy', fontsize='37')
        ax1.set_xlabel(u'Confidence', fontsize='37')

        plt.bar(x, gap_height, bottom=gap_bottom, color='#fb8b05', width=1 / (n_bins), edgecolor='#652b1c', linewidth=1,
                label=u'Gaps', alpha=0.6, hatch='/')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid()

        ax2 = ax1.twinx()  # 组合图
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        x1 = [0, 1]
        y1 = [0, 1]
        ax2.plot(x1, y1, 'k--', ms=10, lw=3, alpha=0.8, label=u'well calibrate')  # 设置线粗细，节点样式
        # plt.legend(loc='upper left')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, fontsize='32')
        plt.show()

        # 绘制平均置信度和精度
        x = np.arange(1 / (2 * 20), 1 + 1 / (2 * 20), 1 / (20))
        fig2, ax1 = plt.subplots(figsize=(11, 11))
        ax1.set_title('MinimalRNN平均置信度和精度', fontproperties='SimSun', fontsize=40)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.bar(x, bin_count_total, color='#6DD5FA', width=1 / (20), edgecolor='#134857', linewidth=1, label=u'Outputs')
        ax1.set_ylabel(u'样本占比%', fontsize='37')
        ax1.set_xlabel(u'Confidence', fontsize='37')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid()

        ax2 = ax1.twinx()  # 组合图
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        x1 = [acc_total, acc_total]
        y1 = [0, 1]
        ax2.plot(x1, y1, 'k--', ms=10, lw=3, alpha=0.8, label=u'Accuracy')  # 设置线粗细，节点样式
        x2 = [confidence_total, confidence_total]
        y2 = [0, 1]
        ax2.plot(x2, y2, 'k:', ms=10, lw=5, alpha=0.8, label=u'Avg confidence')  # 设置线粗细，节点样式
        fig2.legend(loc='upper left',fontsize='32')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.show()

        return ece
    else:
        return ece


def _binary_calibration(target, probs_positive_cls, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)  # 分bins
    binids = np.digitize(probs_positive_cls, bins) - 1  # 返回每个置信度 在bins中的位置 从0开始

    bin_sums = np.bincount(binids, weights=probs_positive_cls, minlength=n_bins)  # 每个bins 置信度的和
    bin_true = np.bincount(binids, weights=target, minlength=n_bins)  # 每个bins 正确的个数（这里是二分类 多分类target应该是 是否正确）
    bin_total = np.bincount(binids, minlength=n_bins)  # 每个bins样本的个数

    nonzero = bin_total != 0  # 防止除0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])  # accuracy
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])  # 平均置信度

    return prob_true, prob_pred, bin_total[nonzero], nonzero


def mask(pred, true):
    """ Drop entries without ground truth data (i.e. NaN in *true*) """
    try:
        index = ~np.isnan(true)
    except Exception:
        print('true', true)
        print('pred', pred)
        raise
    ret = pred[index], true[index]
    assert ret[0].shape[0] == ret[0].shape[0]
    return ret


def parse_data(_ref_frame, _pred_frame, EDL, uncertainty_threshold=1):
    """
    EDL 表示是否是EDL
    uncertainty_threshold表示不确定性阈值 当不确定性小于阈值的 可以参与评估 大于阈值拒识 默认值为1 即对任何样本都评估
    """
    true_label_and_prob = []
    prob_with_mask = []
    alpha_with_mask = []
    uncertainty_with_mask = []
    pred_diag = []
    pred_adas = []
    pred_vent = []
    true_diag = []
    true_adas = []
    true_vent = []

    num_smaple = len(np.unique(_ref_frame.RID))  # 验证集/测试集的样本数

    for i in range(len(_ref_frame)):
        cur_row = _ref_frame.iloc[i]
        subj_data = _pred_frame[_pred_frame.RID == cur_row.RID].reset_index(drop=True)
        dates = subj_data['Forecast Date']
        matched_row = subj_data.iloc[nearest(dates, cur_row.CognitiveAssessmentDate)]

        if EDL:
            uncertainty = matched_row[['uncertainty']].values
            if uncertainty > uncertainty_threshold:  # 不确定性拒识机制
                continue

            alpha = matched_row[[
                'CN relative alpha', 'MCI relative alpha',
                'AD relative alpha'
            ]].values

        prob = matched_row[[
            'CN relative probability', 'MCI relative probability',
            'AD relative probability'
        ]].values

        pred_diag.append(np.argmax(prob))
        pred_adas.append(matched_row['ADAS13'])
        pred_vent.append(subj_data.iloc[nearest(dates, cur_row.ScanDate)]['Ventricles_ICV'])
        true_diag.append(cur_row.Diagnosis)
        true_adas.append(cur_row.ADAS13)
        true_vent.append(cur_row.Ventricles)

        if not np.isnan(cur_row.Diagnosis):
            true_label_and_prob += [(cur_row.Diagnosis, prob)]
            if EDL:
                alpha_with_mask.append(alpha)
                uncertainty_with_mask.append(uncertainty)
            else:
                prob_with_mask.append(prob)

    pred_diag = np.array(pred_diag)
    pred_adas = np.array(pred_adas)
    pred_vent = np.array(pred_vent)
    true_diag = np.array(true_diag)
    true_adas = np.array(true_adas)
    true_vent = np.array(true_vent)

    if EDL:
        alpha_with_mask = np.array(alpha_with_mask, dtype=float)
        uncertainty_with_mask = np.array(uncertainty_with_mask, dtype=float)
    else:
        prob_with_mask = np.array(prob_with_mask, dtype=float)

    pred_diag, true_diag = mask(pred_diag, true_diag)
    pred_adas, true_adas = mask(pred_adas, true_adas)
    pred_vent, true_vent = mask(pred_vent, true_vent)

    if EDL:
        return true_label_and_prob, pred_diag, pred_adas, pred_vent, \
               true_diag, true_adas, true_vent, alpha_with_mask, uncertainty_with_mask, num_smaple
    else:
        return true_label_and_prob, pred_diag, pred_adas, pred_vent, \
               true_diag, true_adas, true_vent, prob_with_mask, num_smaple


def is_date_column(col):
    """ Is the column of type datetime """
    return np.issubdtype(col.dtype, np.datetime64)


def eval_submission(ref_frame, pred_frame, EDL, draw=False, total_epoch=300, uncertainty_threshold=1):
    """ Evaluate mAUC, BCA, ADAS13 MAE, and ventricles MAE """
    assert is_date_column(ref_frame['CognitiveAssessmentDate'])
    assert is_date_column(ref_frame['ScanDate'])
    assert is_date_column(pred_frame['Forecast Date'])

    if EDL:
        true_labels_and_prob, p_diag, p_adas, p_vent, t_diag, t_adas, t_vent, alpha, uncertainty, num_smaple = \
            parse_data(ref_frame, pred_frame, EDL, uncertainty_threshold)
    else:
        true_labels_and_prob, p_diag, p_adas, p_vent, t_diag, t_adas, t_vent, prob, num_smaple = \
            parse_data(ref_frame, pred_frame, EDL)

    if len(p_diag) == 0:
        mauc = float('NaN')
        bca = float('NaN')
    else:
        mauc = MAUC(true_labels_and_prob, no_classes=3)
        bca = calcBCA(p_diag, t_diag.astype(int), no_classes=3)

    # 验证集/测试集分类loss 以及ECE
    if EDL:
        if len(p_diag) == 0:
            ece = float('NaN')
            cls_loss = float('NaN')
        else:
            # ECE误差
            target = np.array(p_diag == t_diag, dtype=int)
            confidence = np.array(1 - uncertainty)  # 置信度是 1-uncertainty
            ece = ece_binary(confidence, target, draw=draw)

            # 分类loss
            alpha = torch.from_numpy(alpha)
            label = torch.from_numpy(t_diag).to(torch.int64)
            cls_loss = evidential_classification(alpha, label, np.log(50) / np.log(100) * total_epoch,
                                                 total_epoch) / num_smaple  # EDL_loss 评估时 KL退火系数直接设置为1    cu两项权重相同

    else:
        # ECE误差
        target = np.array(p_diag == t_diag, dtype=int)
        confidence = prob.max(1)  # 置信度是logits的最大值 也是类标对应的类别概率
        ece = ece_binary(confidence, target, draw=draw)

        # 分类loss
        prob = torch.from_numpy(prob)
        label = torch.from_numpy(t_diag).to(torch.int64)
        cls_loss = F.cross_entropy(prob, label, reduction='sum') / num_smaple

    if len(p_diag) == 0:
        adas = float('NaN')
        vent = float('NaN')
    else:
        adas = np.mean(np.abs(p_adas - t_adas))
        vent = np.mean(np.abs(p_vent - t_vent))

    return {'mAUC': mauc, 'bca': bca, 'adasMAE': adas, 'ventsMAE': vent, 'cls_loss': cls_loss, 'ECE': ece}


def get_arg_parser(i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', '-r', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                                     'Trustworthy_AD/Trustworthy_AD/folds/fold' + str(i) + '_test.csv')
    parser.add_argument('--prediction', '-p', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                                      'Trustworthy_AD/Trustworthy_AD/output/Ours/prediction-ECE-测试' + str(
        i) + '.csv')
    parser.add_argument('--EDL', action='store_true', default=True)  # 是否使用不确定性预测
    return parser


def main():
    mAUC = []
    bca = []
    ECE = []
    adasMAE = []
    ventsMAE = []
    for i in range(20):
        args = get_arg_parser(i).parse_args()
        result = eval_submission(misc.read_csv(args.reference), misc.read_csv(args.prediction), args.EDL, draw=True,
                                 total_epoch=300, uncertainty_threshold=1)  # 评估输出结果

        print('fold', i, '>>')
        print('########### Metrics for clinical status ##################')
        print('mAUC', result['mAUC'], 'bca', result['bca'], 'ECE', result['ECE'])
        print('########### Mean Absolute Error (MAE) ##################')
        print('adasMAE', result['adasMAE'], 'ventsMAE', result['ventsMAE'])
        mAUC.append(result['mAUC'])
        bca.append(result['bca'])
        ECE.append(result['ECE'])
        adasMAE.append(result['adasMAE'])
        ventsMAE.append(result['ventsMAE'])
        print('\n')
    mAUC = np.array(mAUC)
    bca = np.array(bca)
    ECE = np.array(ECE)
    adasMAE = np.array(adasMAE)
    ventsMAE = np.array(ventsMAE)
    print('mAUC平均：', mAUC.mean(), 'mAUC方差：', mAUC.std())
    print('bca平均：', bca.mean(), 'bca方差：', bca.std())
    print('ECE平均：', ECE.mean(), 'ECE方差：', ECE.std())
    print('adasMAE平均：', adasMAE.mean(), 'adasMAE方差：', adasMAE.std())
    print('ventsMAE平均：', ventsMAE.mean(), 'ventsMAE方差：', ventsMAE.std())


if __name__ == '__main__':
    main()
