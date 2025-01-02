from __future__ import print_function
import torch as t
from model import highwayNet
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import utils
## Network Arguments

import loader as lo

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 13
args['out_length'] = 25
args['grid_size'] = (11, 1)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['batch_size'] = 256
args['num_lat_classes'] = 8
args['num_lon_classes'] = 3
args['use_maneuvers'] = True
args['train_flag'] = False
args['use_planning'] = True
args['val_use_mse']=True
# Evaluation metric:
class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    ## Helper function for log sum exp calculation: 一个计算公式
    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=8, num_lon_classes=3,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def main(self, name, val):
        model_step = 1
        # args['train_flag'] = not args['use_maneuvers']
        net = highwayNet(args)
        net.load_state_dict(t.load('trained_models/wave_7.tar'))
        net = net.to(device)
        net.eval()
        if val:
            t2 = lo.roundDataset('/home/paperProject/DiffAttention/data/round/TestSet.mat')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)  # 6716batch
        else:
            # ------------------------------------------------------------
            # a = generator.mapping
            # xx = t.tensor([[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1],
            #                [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1],
            #                [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]], dtype=t.float).permute(1, 0).to(
            #     device)
            # softa = t.cat(t.softmax(t.matmul(a, xx), dim=0).chunk(9, -1), dim=1).squeeze().cpu().detach().numpy()
            # a = t.cat(a.chunk(6, -1), dim=1).squeeze().cpu().detach().numpy()
            # result = np.concatenate((a, softa), axis=-1).transpose()
            # data = pd.DataFrame(result)
            # data.to_excel(writer, name, float_format='%.5f')
            # writer.save()
            t2 = lo.roundDataset('/home/paperProject/DiffAttention/data/round/TestSet.mat')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], num_workers=8,
                                       collate_fn=t2.collate_fn)  # 6716batch
        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0
        lossVals_fde = t.zeros(1).to(device)
        counts_fde = t.zeros(1).to(device)

        lossVals_ade = t.zeros(1).to(device)
        counts_ade = t.zeros(1).to(device)

        counts_mr_2m = t.zeros(1).to(device)
        lossVals_mr_2m = t.zeros(1).to(device)

        counts_mr_5m = t.zeros(1).to(device)
        lossVals_mr_5m = t.zeros(1).to(device)
        val_batch_count = len(valDataloader)
        intention_counts=0

        lossAcc_lon = t.zeros(1).to(device)
        lossPre_lon = t.zeros(1).to(device)
        lossRecall_lon = t.zeros(1).to(device)
        lossF1_lon = t.zeros(1).to(device)

        lossAcc_lat = t.zeros(1).to(device)
        lossPre_lat= t.zeros(1).to(device)
        lossRecall_lat = t.zeros(1).to(device)
        lossF1_lat = t.zeros(1).to(device)

        print("begin.................................", name)
        with(t.no_grad()):
            for idx, data in enumerate(tqdm(valDataloader)):
                hist, nbrs, nbr_list_len, mask, fut, lat_enc, lon_enc, op_mask, edge_index , ve_matrix,ac_matrix,man_matrix,graph_matrix,view_mask = data
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :]
                fut = fut.to(device)
                op_mask = op_mask[:args['out_length'], :, :]
                op_mask = op_mask.to(device)
                te = time.time()
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                all_time += time.time() - te
                nbrsss += 1
                #if nbrsss > args['time']*args['num_worker']:
                #    print(all_time / nbrsss,"ref time")
                if not args['train_flag']:
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):  # 128
                            lat_man = t.argmax(lat_enc[k, :]).detach()
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 8 + lat_man
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        l, c, loss = utils.maskedMSETest(fut_pred_max, fut, op_mask)
                        l_fde, c_fde, loss_fde = utils.FDETest(fut_pred_max, fut, op_mask)
                        l_ade, c_ade, loss_ade = utils.ADETest(fut_pred_max, fut, op_mask)
                        num_mr_2m, c_mr_2m = utils.MRTest(fut_pred_max, fut, op_mask, 2)
                        num_mr_5m, c_mr_5m = utils.MRTest(fut_pred_max, fut, op_mask, 5)
                else:
                    if args['val_use_mse']:
                        l, c, loss = utils.maskedMSETest(fut_pred, fut, op_mask)
                        l_fde, c_fde, loss_fde = utils.FDETest(fut_pred, fut, op_mask)
                        l_ade, c_ade, loss_ade = utils.ADETest(fut_pred, fut, op_mask)
                        num_mr_2m, c_mr_2m = utils.MRTest(fut_pred, fut, op_mask, 2)
                        num_mr_5m, c_mr_5m = utils.MRTest(fut_pred, fut, op_mask, 5)
                lossVals += l.detach()
                counts += c.detach()
                lossVals_mr_2m += num_mr_2m
                lossVals_mr_5m += num_mr_5m

                avg_val_loss += loss.item()
                counts_fde += c_fde.detach()
                lossVals_fde += l_fde.detach()

                counts_ade += c_ade.detach()
                lossVals_ade += l_ade.detach()
                counts_mr_2m += c_mr_2m
                counts_mr_5m += c_mr_5m
                avg_val_loss += loss.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            # tqdm.write('valmse:', avg_val_loss / val_batch_count)
            if args['val_use_mse']:
                rmseOverall = (t.pow(lossVals / counts, 0.5)).cpu()
                pred_rmse_horiz = utils.horiz_eval(rmseOverall, 4)
                loss_metrics = {
                    'valmse': avg_val_loss / val_batch_count,
                    'FDE(m)': lossVals_fde / counts_fde,
                    'ADE(m)': lossVals_ade / counts_ade,
                    'MR(2m)': lossVals_mr_2m / counts_mr_2m,
                    'MR(5m)': lossVals_mr_5m / counts_mr_5m,
                    'ref time': all_time / nbrsss,
                    'RMSE': pred_rmse_horiz,
                }
                for metric, value in loss_metrics.items():
                    print(f'{metric}\t=> {value}')

            else:
                print('valnll:', avg_val_loss / val_batch_count)
                print(lossVals / counts)
            # print(lossVals/counts*0.3048)


if __name__ == '__main__':
    names = ['wave_7']
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(name=epoch, val=False)


