import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import misc
from evaluation import eval_submission
from evaluation import parse_data

plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimSun', style='white', )  # 解决Seaborn中文显示问题


def get_arg_parser(i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', '-r', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                                     'Trustworthy_AD/Trustworthy_AD/folds/fold' + str(i) + '_test.csv')
    parser.add_argument('--prediction', '-p', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/'
                                                      'Trustworthy_AD/Trustworthy_AD/output/Ours/prediction-CU-测试'
                                                      + str(i) + '.csv')
    parser.add_argument('--EDL', action='store_true', default=True)  # 是否使用不确定性预测
    return parser


def BCA_mAUC_uncertainty_threshold(i, step):
    '''
    不确定性阈值和分类性能曲线
    i表示需要画第几个模型的曲线
    step表示步长
    '''
    args = get_arg_parser(i).parse_args()
    mAUC_list = []
    BCA_list = []
    x = np.linspace(0.15, 1, step + 1)

    for j in np.linspace(0.15, 1, step + 1):
        result = eval_submission(misc.read_csv(args.reference), misc.read_csv(args.prediction), args.EDL, draw=False,
                                 total_epoch=300, uncertainty_threshold=j)
        mAUC_list.append(result['mAUC'])
        BCA_list.append(result['bca'])

    mAUC_list = np.array(mAUC_list)
    BCA_list = np.array(BCA_list)

    # 绘制曲线mAUC
    fig1 = plt.figure(figsize=(6,6))
    ax1 = fig1.add_subplot(111)
    # ax1.set_title('', fontproperties='SimHei', fontsize=20)
    ax1.set_xlim([0.1, 1.05])
    ax1.set_ylim([0.925, 1])
    ax1.plot(x, mAUC_list, color='k', label=u'mAUC', marker='o', linewidth=5, markersize=12)
    ax1.set_ylabel(u'mAUC', fontsize='35')
    ax1.set_xlabel(u'不确定性拒识阈值', fontsize='35')
    plt.grid()
    plt.legend(loc='upper right',fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

    # 绘制BCA曲线
    fig2 = plt.figure(figsize=(6, 6))
    ax1 = fig2.add_subplot(111)
    # ax1.set_title('', fontproperties='SimHei', fontsize=20)
    ax1.set_xlim([0.1, 1.05])
    ax1.set_ylim([0.85, 1])
    ax1.plot(x, BCA_list, color='k', label=u'BCA', marker='o', linewidth=5, markersize=12)
    ax1.set_ylabel(u'BCA', fontsize='35')
    ax1.set_xlabel(u'不确定性拒识阈值', fontsize='35')
    plt.grid()
    plt.legend(loc='upper right', fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

    return


def mAUC_BCA_plot():
    '''
    绘制不同模型的mAUC_BCA柱状图
    '''
    # mAUC
    x = ['MiniRNN', 'GRU', 'LSTM', 'LSS', 'SVM', 'Ours']
    error_params = dict(elinewidth=4, ecolor='k', capsize=5)  # 误差样式
    color = ['#6DD5FA', '#ee9ca7', '#38ef7d', '#f7b733', '#6A82FB','#74ebd5']  # 颜色

    mAUC = [0.943, 0.927, 0.925, 0.926, 0.929, 0.943]
    std = [0.014, 0.014, 0.019, 0.025, 0.013, 0.017]  # 方差
    fig1 = plt.figure(figsize=(11, 11))
    ax1 = fig1.add_subplot(111)
    ax1.set_ylim([0.85, 0.97])
    ax1.bar(x, mAUC, color=color, linewidth=1, yerr=std, error_kw=error_params)
    ax1.set_ylabel(u'诊断mAUC', fontsize='37')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.show()

    # BCA
    BCA = [0.884, 0.874, 0.866, 0.861,0.841,0.88]
    std = [0.023, 0.015, 0.025, 0.029,0.023,0.022]
    fig1 = plt.figure(figsize=(11, 11))
    ax1 = fig1.add_subplot(111)
    ax1.set_ylim([0.8, 0.93])
    ax1.bar(x, BCA, color=color, linewidth=1, yerr=std, error_kw=error_params)
    ax1.set_ylabel(u'诊断BCA', fontsize='37')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.show()

    return


def cls_confidence_acc_EDL():
    '''
    测试集所有样本 每个类别的 平均错误率 不确定性 错误不确定性
    '''
    pred = []
    true = []
    u = []
    CN_pred = []
    CN_true = []
    CN_u = []
    CN_confi = []
    MCI_pred = []
    MCI_true = []
    MCI_u = []
    MCI_confi = []
    AD_pred = []
    AD_true = []
    AD_u = []
    AD_confi = []
    for i in range(20):
        args = get_arg_parser(i).parse_args()
        _, p_diag, _, _, t_diag, _, _, _, uncertainty, _ = \
            parse_data(misc.read_csv(args.reference), misc.read_csv(args.prediction), True, uncertainty_threshold=1)
        t_diag = t_diag.astype(int)

        for j in range(len(t_diag)):
            pred.append(p_diag[j])
            true.append(t_diag[j])
            u.append(uncertainty[j])
            if t_diag[j] == 0:
                CN_true.append(t_diag[j])
                CN_pred.append(p_diag[j])
                CN_u.append(uncertainty[j])
                CN_confi.append(1-uncertainty[j])
            elif t_diag[j] == 1:
                MCI_true.append(t_diag[j])
                MCI_pred.append(p_diag[j])
                MCI_u.append(uncertainty[j])
                MCI_confi.append(1 - uncertainty[j])
            else:
                AD_true.append(t_diag[j])
                AD_pred.append(p_diag[j])
                AD_u.append(uncertainty[j])
                AD_confi.append(1 - uncertainty[j])

    pred = np.array(pred)
    true = np.array(true)
    u = np.array(u)
    CN_pred = np.array(CN_pred)
    CN_true = np.array(CN_true)
    CN_u = np.array(CN_u)
    CN_confi = np.array(CN_confi)
    MCI_pred = np.array(MCI_pred)
    MCI_true = np.array(MCI_true)
    MCI_u = np.array(MCI_u)
    MCI_confi = np.array(MCI_confi)
    AD_pred = np.array(AD_pred)
    AD_true = np.array(AD_true)
    AD_u = np.array(AD_u)
    AD_confi = np.array(AD_confi)

    error = 1-np.array(pred==true).sum()/len(true)
    acc = np.array(pred==true).sum()/len(true)
    mean_ErrU = np.mean(u[pred != true])
    mean_RightU = np.mean(u[pred == true])
    print("总错误率:",error,"   错误不确定性：",mean_ErrU,"    总精度：",acc," 正确不确定性：",mean_RightU)

    CN_acc = np.array(CN_pred == CN_true).sum() / len(CN_true)
    CN_meanConfi = np.mean(CN_confi[CN_pred != CN_true])
    CN_error = 1 - np.array(CN_pred == CN_true).sum() / len(CN_true)
    CN_meanErrU = np.mean(CN_u[CN_pred != CN_true])
    CN_meanRightU = np.mean(CN_u[CN_pred == CN_true])
    CN_meanU = np.mean(CN_u)

    MCI_acc = np.array(MCI_pred == MCI_true).sum() / len(MCI_true)
    MCI_meanConfi = np.mean(MCI_confi[MCI_pred != MCI_true])
    MCI_error = 1 - np.array(MCI_pred == MCI_true).sum() / len(MCI_true)
    MCI_meanErrU = np.mean(MCI_u[MCI_pred != MCI_true])
    MCI_meanRightU = np.mean(MCI_u[MCI_pred == MCI_true])
    MCI_meanU = np.mean(MCI_u)

    AD_acc = np.array(AD_pred == AD_true).sum() / len(AD_true)
    AD_meanConfi = np.mean(AD_confi[AD_pred != AD_true])
    AD_error = 1 - np.array(AD_pred == AD_true).sum() / len(AD_true)
    AD_meanErrU = np.mean(AD_u[AD_pred != AD_true])
    AD_meanRightU = np.mean(AD_u[AD_pred == AD_true])
    AD_meanU = np.mean(AD_u)

    # 画图
    x = ['CN', 'MCI', 'AD']
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title('本文方法', fontproperties='SimSun', fontsize=35)
    ax1.set_ylim([0, 0.3])
    ax1.bar(x, [CN_error, MCI_error, AD_error], color='k', linewidth=1, label=u'错误率', alpha=0.3)
    ax1.set_ylabel(u'错误率', fontsize='35')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=20)

    ax3 = ax1.twinx()  # 组合图
    ax3.set_ylim([0, 0.7])
    ax3.plot(x, [CN_meanU, MCI_meanU, AD_meanU], 'k', ms=15, lw=5, marker='o', label=u'平均不确定性')
    ax3.plot(x, [CN_meanErrU, MCI_meanErrU, AD_meanErrU], 'k', ms=15, lw=5, marker='o', label=u'错误平均不确定性')
    ax3.plot(x, [CN_meanRightU, MCI_meanRightU, AD_meanRightU], 'k', ms=15, lw=5, marker='^', label=u'正确平均不确定性')
    ax3.set_ylabel(u'平均不确定性', fontsize='35', rotation=90)

    plt.grid()
    fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, fontsize='30')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=20)
    plt.show()

    # fig, ax1 = plt.subplots(figsize=(6, 6))
    # ax1.set_title('本文方法', fontproperties='SimSun', fontsize=35)
    # ax1.set_ylim([0.7, 1])
    # ax1.bar(x, [CN_acc, MCI_acc, AD_acc], color='k', linewidth=1, label=u'精度', alpha=0.3)
    # ax1.set_ylabel(u'精度', fontsize='35')
    # plt.xticks(fontsize=35)
    # plt.yticks(fontsize=20)
    #
    # ax3 = ax1.twinx()  # 组合图
    # ax3.set_ylim([0.35, 0.85])
    # ax3.plot(x, [CN_meanConfi, MCI_meanConfi, AD_meanConfi], 'k', ms=15, lw=5, marker='o', label=u'可靠性')
    # ax3.set_ylabel(u'可靠性', fontsize='35', rotation=90)
    #
    # plt.grid()
    # fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, fontsize='30')
    # plt.xticks(fontsize=35)
    # plt.yticks(fontsize=20)
    # plt.show()
    return


def cls_confidence_acc():
    '''
    测试集所有样本 每个类别的 平均错误率 不确定性 错误不确定性
    '''
    CN_pred = []
    CN_true = []
    CN_confi = []
    MCI_pred = []
    MCI_true = []
    MCI_confi = []
    AD_pred = []
    AD_true = []
    AD_confi = []
    for i in range(20):
        args = get_arg_parser(i).parse_args()
        _, p_diag, _, _, t_diag, _, _, prob, _ = \
            parse_data(misc.read_csv(args.reference), misc.read_csv(args.prediction), False, uncertainty_threshold=1)
        t_diag = t_diag.astype(int)
        prob=prob.max(1)        # 取最大的

        for j in range(len(t_diag)):
            if t_diag[j] == 0:
                CN_true.append(t_diag[j])
                CN_pred.append(p_diag[j])
                CN_confi.append(prob[j])
            elif t_diag[j] == 1:
                MCI_true.append(t_diag[j])
                MCI_pred.append(p_diag[j])
                MCI_confi.append(prob[j])
            else:
                AD_true.append(t_diag[j])
                AD_pred.append(p_diag[j])
                AD_confi.append(prob[j])

    CN_pred = np.array(CN_pred)
    CN_true = np.array(CN_true)
    CN_confi = np.array(CN_confi)
    MCI_pred = np.array(MCI_pred)
    MCI_true = np.array(MCI_true)
    MCI_confi = np.array(MCI_confi)
    AD_pred = np.array(AD_pred)
    AD_true = np.array(AD_true)
    AD_confi = np.array(AD_confi)

    CN_acc = np.array(CN_pred == CN_true).sum() / len(CN_true)
    # CN_error = 1 - np.array(CN_pred == CN_true).sum() / len(CN_true)
    CN_meanConfi = np.mean(CN_confi[CN_pred != CN_true])

    MCI_acc = np.array(MCI_pred == MCI_true).sum() / len(MCI_true)
    # MCI_error = 1 - np.array(MCI_pred == MCI_true).sum() / len(MCI_true)
    MCI_meanConfi = np.mean(MCI_confi[MCI_pred != MCI_true])

    AD_acc = np.array(AD_pred == AD_true).sum() / len(AD_true)
    AD_meanConfi = np.mean(AD_confi[AD_pred != AD_true])

    # 画图
    x = ['CN', 'MCI', 'AD']
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title('MinimalRNN', fontproperties='SimSun', fontsize=35)
    ax1.set_ylim([0.7, 1])
    ax1.bar(x, [CN_acc, MCI_acc, AD_acc], color='k', linewidth=1, label=u'精度', alpha=0.3)
    ax1.set_ylabel(u'精度', fontsize='35')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=20)

    ax3 = ax1.twinx()  # 组合图
    ax3.set_ylim([0.7, 1])
    ax3.plot(x, [CN_meanConfi, MCI_meanConfi, AD_meanConfi], 'k', ms=15, lw=5, marker='^',label=u'可靠性')
    ax3.set_ylabel(u'可靠性', fontsize='35', rotation=90)
    # ax3.plot(x, [CN_meanErr1U, MCI_meanErr1U, AD_meanErr1U], 'r', ms=10, lw=3, marker='^', label=u'1# error uncertainty')

    # ax3.plot(x, [CN_meanU, MCI_meanU, AD_meanU], 'k', ms=10, lw=2, marker='o', label=u'2# uncertainty')
    # ax3.plot(x, [CN_mean1U, MCI_mean1U, AD_mean1U], 'k', ms=10, lw=2, marker='^', label=u'1# uncertainty')

    plt.grid()
    fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, fontsize='30')
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=20)
    plt.show()
    return


# BCA_mAUC_uncertainty_threshold(1, 17)
# mAUC_BCA_plot()
cls_confidence_acc_EDL()
# cls_confidence_acc()
