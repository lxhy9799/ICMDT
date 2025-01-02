import os
import time
import matplotlib
import matplotlib.lines as mlines
import numpy as np
from loader import roundDataset
from model_round import Net
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch as t
import matplotlib.pyplot as plt
import utils

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
## Network Arguments

parser = argparse.ArgumentParser(description='Evaluating:')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default=True)
parser.add_argument('--use_true_man', action="store_false", help='(default: True)',
                    default=True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)', default=128)
parser.add_argument('--tensorboard', action="store_true", help='if use tensorboard (default: True)', default=True)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int, help='default: (13,3)', nargs=2, default=[11, 1])
parser.add_argument('--in_length', type=int, help='History sequence (default: 16)',
                    default=13)  # 3s history traj at 5Hz
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 25)',
                    default=25)  # 5s future traj at 5Hz
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors', default=8)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors', default=3)
# Network hyperparameters------------------------------------------
# ----------------------------------------------------
parser.add_argument('--num_features', type=int, help='The last dimension of input', default=3)
parser.add_argument('--num_blocks', type=int, default=1, help='')
parser.add_argument('--input_embed_size', type=int, default=32, help='embed size')
parser.add_argument('--lstm_encoder_size', type=int, default=64, help='-dimension of input')
parser.add_argument('--att_out_size', type=int, default=32, help='dimension of attention')
parser.add_argument('--ff_hidden_size', type=int, default=128, help='')
parser.add_argument('--decoder_size', type=int, default=64, help='LSTM size')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention head')
parser.add_argument('--train_flag', type=bool, default=False, help='train flag')
parser.add_argument('--use_maneuvers', type=bool, default=True, help='')
# Training setting------------------------------------------
parser.add_argument('--name', type=str, help='log name', default="round")
parser.add_argument('--test_set', type=str, default='data/round/TestSet.mat', help='Path to test datasets')
parser.add_argument('--val_set', type=str, default='data/round/ValSet.mat', help='Path to validation datasets')
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--pretrain_epochs', type=int, help='epochs of pre-training using MSE', default=6)
parser.add_argument('--train_epochs', type=int, default=3)
parser.add_argument('--dataset_name', type=str, default='round')
parser.add_argument('--val_use_mse', type=bool, default=True, help='')
parser.add_argument('--experimental_name', type=str, help='', default='test')
parser.add_argument('--ablation_study', type=bool, default=False, help='')

net_args = parser.parse_args()


class Evaluate():
    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1
        self.delta = 2.0
        self.v_w = 4
        self.v_l = 8

    def main(self, val):
        model_step = 1
        net = Net(net_args)
        check_point = t.load('./trained_models/%s/%s-pre%d-nll%d_%s.pt' % (
            net_args.dataset_name, net_args.name, net_args.pretrain_epochs, net_args.train_epochs,
            net_args.experimental_name))
        net.load_state_dict(check_point['model_state_dict'])
        net = net.to(device)
        net.eval()
        if net_args.dataset_name == "round":
            if net_args.ablation_study:
                t2 = roundDataset(net_args.val_set)
            else:
                t2 = roundDataset(net_args.test_set)
        valDataloader = DataLoader(t2, batch_size=net_args.batch_size, shuffle=False,
                                   num_workers=net_args.num_workers,
                                   collate_fn=t2.collate_fn, drop_last=True)
        lossVals = t.zeros(net_args.out_length).to(device)
        counts = t.zeros(net_args.out_length).to(device)
        avg_val_loss = 0
        lossVals_fde = t.zeros(1).to(device)
        counts_fde = t.zeros(1).to(device)

        lossVals_ade = t.zeros(1).to(device)
        counts_ade = t.zeros(1).to(device)

        counts_mr_2m = t.zeros(1).to(device)
        lossVals_mr_2m = t.zeros(1).to(device)

        counts_mr_5m = t.zeros(1).to(device)
        lossVals_mr_5m = t.zeros(1).to(device)
        all_time = 0
        nbrsss = 0

        val_batch_count = len(valDataloader)
        print("begin.................................\n")
        with(t.no_grad()):
            for idx, data in enumerate(tqdm(valDataloader)):
                hist, nbrs, nbr_list_len, mask, fut, lat_enc, lon_enc, op_mask, edge_index, ve_matrix, ac_matrix, man_matrix, graph_matrix, view_mask = data
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                nbr_list_len = nbr_list_len.cuda()
                fut = fut.cuda()
                mask = mask.cuda()
                op_mask = op_mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                edge_index = edge_index.to(device)
                ve_matrix = ve_matrix.to(device)
                ac_matrix = ac_matrix.to(device)
                man_matrix = man_matrix.to(device)
                graph_matrix = graph_matrix.to(device)
                view_mask = view_mask.to(device)
                # hist= self.simulate_latency(hist,2)
                # Forward pass
                te = time.time()
                fut_pred, pi, lat_pred, lon_pred, _ = net(hist, nbrs, mask, lat_enc, lon_enc, edge_index, ve_matrix,
                                                          ac_matrix, man_matrix, graph_matrix)
                all_time += time.time() - te
                nbrsss += 1
                # indices = []
                # fut_pred_max = t.zeros_like(fut_pred[0])
                # for k in range(lat_pred.shape[0]):
                #     lat_man = t.argmax(lat_enc[k, :]).detach()
                #     lon_man = t.argmax(lon_enc[k, :]).detach()
                #     index = lon_man * 8 + lat_man
                #     indices.append(index)
                #     fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                fut_ = fut.permute(1, 0, 2)
                l2_norm = t.norm(fut_pred[..., :2] - fut_[..., :2].unsqueeze(1), dim=-1).sum(dim=-1)
                best_mode = t.argmin(l2_norm, dim=-1)
                fut_pred_max = fut_pred[t.arange(fut_pred.shape[0]), best_mode]
                fut_pred_max = fut_pred_max.permute(1, 0, 2)
                l, c, loss = utils.maskedMSETest(fut_pred_max, fut, op_mask)
                l_fde, c_fde, loss_fde = utils.FDETest(fut_pred_max, fut, op_mask)
                l_ade, c_ade, loss_ade = utils.ADETest(fut_pred_max, fut, op_mask)
                num_mr_2m, c_mr_2m = utils.MRTest(fut_pred_max, fut, op_mask, 2)
                num_mr_5m, c_mr_5m = utils.MRTest(fut_pred_max, fut, op_mask, 5)
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
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            if net_args.val_use_mse:
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
                    'RMSE_mean': pred_rmse_horiz.mean(),
                }
                for metric, value in loss_metrics.items():
                    print(f'{metric}\t=> {value}')

if __name__ == '__main__':
    evaluate = Evaluate()
    evaluate.drawImg = False
    evaluate.main(val=False)
