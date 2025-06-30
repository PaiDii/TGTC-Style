import torch
import torch.nn as nn

# 计算均均值和标准差
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# AdaIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    # 归一化
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    # 感觉相当于把风格标准差当作权重，均值当作偏移
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


def styleLoss(input, target):
    ib, ic, ih, iw = input.size()
    iF = input.view(ib, ic, -1)
    iMean = torch.mean(iF, dim=2)
    iCov = GramMatrix(input)

    tb, tc, th, tw = target.size()
    tF = target.view(tb, tc, -1)
    tMean = torch.mean(tF, dim=2)
    tCov = GramMatrix(target)

    loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov, tCov)
    # 归一化：除以什么变量相当于对什么变量进行归一化，消除变量变化所带来的影响
    # 当无法确保不同图像下该变量个数是否一致，则可以除一下
    return loss/tb

# Gram矩阵被用于表征图像的风格。在图像修复问题中，很常用的一项损失叫做风格损失（style loss），风格损失正是基于预测结果和真值之间的Gram矩阵的差异构建的。
# bmm == 3dim, mm == 2dim
def GramMatrix(input):
    b, c, h, w = input.size()
    f = input.view(b, c, h*w) # bxcx(hxw)
    # torch.bmm(batch1, batch2, out=None)
    # batch1: bxmxp, batch2: bxpxn -> bxmxn
    G = torch.bmm(f, f.transpose(1, 2))  # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
    # 归一化 Gram 矩阵可以消除特征图大小和通道数的影响，使得不同大小和通道数的特征图之间可以进行比较。
    # 通过将结果除以 c x h x w，相当于将每个元素除以特征图的总大小，从而将值范围缩放到一个相对较小的区间。
    return G.div_(c*h*w)
