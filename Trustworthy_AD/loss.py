import torch
import torch.nn.functional as F

'''分类问题Dirichlet分布的相关损失函数'''
def dirichlet_kl(alpha, y):
    # KL散度项（用于膨胀错误不确定性）
    alpha = y + (1 - y) * alpha                     # 计算去除正确分类的证据  现在alpha为不正确的证据

    beta = torch.ones_like(alpha)                   # 不正确的部分组成的新alpha的分布 应该和参数全为1的Dirichlet很接近

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()     # 计算KL散度 公式第一项分子 和 分母第一项
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta                               # 公式第二项
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.sum()


def dirichlet_mse(alpha, y):
    # 贝叶斯risk项（用于拟合正确证据）
    # alpha为分类网络输出的Dirichlet分布参数alpha   y为类标 one-hot编码
    sum_alpha = alpha.sum(-1, keepdims=True)        # alpha形状  batch*K类
    p = alpha / sum_alpha                           # p为每个类别Dirichlet的期望
    t1 = (y - p).pow(2).sum(-1)                     # t1为k个类别mse的和
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)  # t2为Dirichlet分布的方差
    mse = t1 + t2
    return mse.sum()


def dirichlet_cu(alpha, label, num_class, current_epoch, total_epoch):
    # 不确定性校准项

    annealing_start = torch.tensor(0.314, dtype=torch.float32)       # 0.01在85%epoch两项权重相等   0.25在50%epoch权重相等   0.314 40%    0.177在%60
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * current_epoch)      # 退火系数

    S = torch.sum(alpha, dim=1, keepdim=True)  # 计算狄利克雷强度
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)       # alpha的最大值就是预测的类别  pred_scores为类别概率  pred_cls为类标
    uncertainty = num_class / S     # 不确定性

    target = label.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))     # 标记当前分类是否正确

    acc_uncertain = - pred_scores * torch.log(1 - uncertainty + 1e-10)      # 分类正确  不确定性要低
    inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + 1e-10)    # 分类错误  不确定性要高

    CU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain

    return CU.sum()


def evidential_classification(alpha, y, current_epoch, total_epoch):
    # 分类问题总损失函数
    # alpha为分类网络输出的Dirichlet分布参数alpha   y为类标类别索引（未one-hot编码）   lamb为退火系数
    num_classes = alpha.shape[-1]
    label = y       # 类标 未one-hot
    y = F.one_hot(y.to(torch.int64), num_classes)                   # one-hot编码

    annealing_coef = torch.min(  # 退火系数的计算
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(current_epoch / 10, dtype=torch.float32),
    )

    # 分类问题总损失函数是 贝叶斯risk项（用于拟合正确证据）+KL散度项（用于膨胀错误不确定性）+不确定性校准函数
    return dirichlet_mse(alpha, y) + annealing_coef * dirichlet_kl(alpha, y) + dirichlet_cu(alpha, label, num_classes, current_epoch, total_epoch)