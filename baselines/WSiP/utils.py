from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from scipy.optimize import curve_fit



## Quintic spline definition.
# def quintic_spline(x, z, a, b, c, d, e):
#     return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5
def quintic_spline(x, z, a, b, c):
    return z + a * x + b * x ** 2 + c * x ** 3


def outputActivation(x, displacement = False):
    if displacement:
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    muX = x[..., 0:1]
    muY = x[..., 1:2]
    sigX = x[..., 2:3]
    sigY = x[..., 3:4]
    rho = x[..., 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=-1)
    return out


def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * (
                torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                            2) - 2 * rho * torch.pow(
            sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True,
                  avg_along_time=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]  # x轴的均值
                muY = y_pred[:, :, 1]  # y轴的均值
                sigX = y_pred[:, :, 2]  # x轴的方差
                sigY = y_pred[:, :, 3]  # y轴的方差
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]

                out = -(0.5 * torch.pow(ohr, 2) * (
                            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                        2) - 2 * rho * torch.pow(
                        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(
                    sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc, dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = 0.5 * torch.pow(ohr, 2) * (
                    torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                2) - 2 * rho * torch.pow(
                sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:, :, 0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts

def FDETest(y_pred,y_gt,mask):
    acc = torch.zeros_like(mask[-1,:,:])
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    out = torch.pow(out,0.5)
    acc[:, 0] = out
    acc[:, 1] = out
    acc = acc * mask[-1,:,:]
    lossVal = torch.sum(acc[:, 0], dim=0)
    counts = torch.sum(mask[-1,:, 0], dim=0)
    loss = torch.sum(acc) / torch.sum(mask)
    return lossVal, counts, loss

def ADETest(y_pred,y_gt,mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    out=  torch.pow(out,0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    lossVal = torch.sum(lossVal,dim=0)

    counts = torch.sum(mask[:, :, 0], dim=1)
    counts = torch.sum(counts,dim=0)
    loss = torch.sum(acc) / torch.sum(mask)
    return lossVal, counts, loss

def MRTest(y_pred,y_gt,mask,threshold=16.4):
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    out=  torch.pow(out,0.5)
    n=0
    for i in range(out.shape[0]):
        if out[i]>threshold:
            n+=1
    counts = mask.shape[1]
    return n, counts

def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all // n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames * i

        if i == n_horiz - 1:
            en_id = n_all - 1
        else:
            en_id = n_frames * i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id + 1])

    return avg_res


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    loss = torch.sum(acc) / torch.sum(mask)
    return lossVal, counts, loss


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
