from __future__ import print_function, division
from scipy import spatial
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
import time
from scipy.interpolate import interp1d


# Dataset class for the rounD dataset
class roundDataset(Dataset):

    def __init__(self, mat_file, t_h=50, t_f=100, d_s=4,
                 enc_size=64,  lat_dim=8,
                 lon_dim=3):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.A = scp.loadmat(mat_file)['anchor_traj_raw']

        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

        self.enc_size = enc_size  # size of encoder LSTM
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.ip_dim= 3
        # self.goal_dim = goal_dim
        # self.en_ex_dim = en_ex_dim
    def __len__(self):
        return len(self.D)


    def __getitem__(self, idx):
        # print('getitem is called ')
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,15:] #14 if no entry_exit_class 15 if there
        neighbors = []

        # Encoding of Lateral and Longitudinal Intention Classes
        lat_class = self.D[idx, 12] - 1
        lat_enc = np.zeros([self.lat_dim])
        lat_enc[int(lat_class)] = 1

        lon_class = self.D[idx, 13] - 1
        lon_enc = np.zeros([self.lon_dim])
        lon_enc[int(lon_class)] = 1

        #Normal==>1  Brake == > 2  ACC == > 2

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        fut= self.getFuture(vehId, t, dsId, lat_class, lon_class)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))
        xy_list = self.get_xy(dsId, t, grid, vehId)
        edge_index = self.graph_xy(xy_list)
        va_list = self.get_va(dsId, t, grid, vehId)
        ve_matrix = self.graph_ve(va_list)
        ac_matrix = self.graph_ac(va_list)
        man_list = self.get_man(dsId, t, grid, vehId)
        man_matrix = self.graph_man(man_list)
        view_grip = self.mask_view(dsId, t, grid, vehId).reshape(1,11)

        return hist, fut, neighbors, lat_enc, lon_enc, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip


    def mask_view(self, dsId, t, grid, vehId):

        view_matrix = np.zeros((11))
        if vehId == 0:
            view_matrix1 = np.array(view_matrix)
            return view_matrix1.reshape(1, 11)
        else:
            if self.T.shape[1] <= vehId - 1:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            velocity = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0 or grid.size == 0:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(1, 11)
            else:
                if velocity < 4:
                    indices = torch.tensor([3, 4, 6, 7])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(1, 11)
                elif velocity >= 4 and velocity <= 8:
                    indices = torch.tensor([1, 2,  7,8, 9])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(1, 11)
                elif velocity > 8:
                    indices = torch.tensor([0, 1, 9,10])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(1, 11)

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()


            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1


                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    # Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, self.ip_dim])
        else:
            veh_tracks = self.T

            if veh_tracks.shape[1] <= vehId - 1:
                return np.empty([0, self.ip_dim])
            refTrack = veh_tracks[dsId - 1][refVehId - 1].transpose()
            vehTrack = veh_tracks[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:self.ip_dim + 1]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, self.ip_dim])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, self.ip_dim])
            return hist

    # Helper function to get track future
    def getFuture(self, vehId, t, dsId, lat_class, lon_class):

        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:self.ip_dim + 1]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos
        # anchor_traj = self.A[int(lon_class), int(lat_class)]
        # anchor_traj = anchor_traj[0:-1:self.d_s, :]
        #
        # fut_anchored = anchor_traj[0:len(fut), :] - fut

        return fut



    def get_man(self, dsId, t, grid, vehId):
        man_list = np.full((len(grid), 2), 0)
        grid[5] = vehId
        refMAN = np.zeros([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refMAN = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refMAN = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refMAN = vehTrack[np.where(vehTrack[:, 0] == t)][0, 10:12]
            if refMAN.size != 0:
                man_list[i] = refMAN.flatten()
        return man_list



    def graph_man(self, man_list):

        man_list = man_list.astype(float)
        man_list[np.where(man_list == 0)] = np.nan
        man_list[np.isinf(man_list)] = np.nan
        max_num_object = 11
        man_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(11):
            for j in range(i + 1, 11):
                if (man_list[i][0] == man_list[j][0]) and (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 2

                elif (man_list[i][0] == man_list[j][0]) or (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 1

        return man_matrix
    def get_va(self, dsId, t, grid, vehId):
        va_list = np.full((len(grid), 2), 0)
        grid[5] = vehId
        refVA = np.zeros([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refVA = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refVA = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refVA = vehTrack[np.where(vehTrack[:, 0] == t)][0, 4:8]

            if refVA.size != 0:
                vx, vy, ax,ay = refVA[0], refVA[1], refVA[2],refVA[3]
                v_combined = np.sqrt(vx ** 2 + vy ** 2)
                a_combined = np.sqrt(ax ** 2 + ay ** 2)
                # va_list[i] = refVA.flatten()
                va_list[i] = [v_combined, a_combined]
        return va_list

    def graph_ve(self, va_list):  #速度

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 11
        ve_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(11):
            for j in range(i + 1, 11):
                ve_matrix[i][j] = (va_list[i][0] - va_list[j][0])
                ve_matrix[j][i] = -ve_matrix[i][j]

        ve_matrix = torch.tensor(ve_matrix).float()
        return ve_matrix
    def get_xy(self, dsId, t, grid, vehId):
        xy_list = np.full((len(grid), 2), np.inf)
        grid[5] = vehId
        refPos= np.empty([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refPos = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refPos = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()

                    if vehTrack.size != 0:
                        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
            if refPos.size != 0:
                xy_list[i] = refPos.flatten()
        return xy_list

    def graph_xy(self, xy_list):

        node1 = []
        node2 = []

        xy_list = xy_list.astype(float)
        max_num_object =11

        neighbor_matrix = np.zeros((max_num_object, max_num_object))

        dist_xy = spatial.distance.cdist(xy_list, xy_list)

        dist_xy[np.isinf(dist_xy)] = np.inf
        dist_xy[np.isnan(dist_xy)] = np.inf

        for i in range(11):
            for j in range(i + 1, 11):
                if dist_xy[i][j] <= 30:
                    node1.append(i)
                    node2.append(j)

        node1 = torch.tensor(node1).unsqueeze(0)
        node2 = torch.tensor(node2).unsqueeze(0)
        edge_index = torch.cat((node1, node2), dim=0)

        return edge_index
    def graph_ac(self, va_list): #加速度

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 11
        ac_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(11):
            for j in range(i + 1, 11):
                ac_matrix[i][j] = (va_list[i][1] - va_list[j][1])
                ac_matrix[j][i] = -ac_matrix[i][j]

        ac_matrix = torch.tensor(ac_matrix).float()
        return ac_matrix

    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        # nbr_batch_size = 0
        nbr_batch_size = 0
        nbr_list_len = torch.zeros(len(samples),1)
        for sample_id , (_, _, nbrs, _, _, _, _, _,_,_) in enumerate(samples):
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
            nbr_list_len[sample_id] = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])

        # nbr_batch_size = int((sum(nbr_list_len)).item())
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, self.ip_dim)

        # Initialize social mask batch:

        mask_batch = torch.zeros(len(samples), 11, self.enc_size)  # (batch,9,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), self.ip_dim)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        ds_ids_batch = torch.zeros(len(samples), 1)
        vehicle_ids_batch = torch.zeros(len(samples), 1)
        frame_ids_batch = torch.zeros(len(samples), 1)
        lat_enc_batch = torch.zeros(len(samples), self.lat_dim)
        lon_enc_batch = torch.zeros(len(samples), self.lon_dim)
        fut_anchored_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)

        edge_index_number = 0
        for sampleId, (_, _, nbrs, _, _, edge_index, _, _,_,_ ) in enumerate(
                samples):
            edge_index_number += edge_index.shape[1]
        edge_index_batch = torch.zeros(len(samples), 2, edge_index_number)
        ac_matrix_batch = torch.zeros(len(samples), 11, 11)
        ve_matrix_batch = torch.zeros(len(samples), 11, 11)
        man_matrix_batch = torch.zeros(len(samples), 11, 11)
        view_mask_batch = torch.zeros(len(samples), 1, 11)

        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0

        for sampleId, (hist, fut, nbrs,lat_enc, lon_enc, edge_index, ve_matrix, ac_matrix, man_matrix, view_mask) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            # if np.argmax(lon_enc)!=2 :
            #     continue
            for k in range(self.ip_dim):
                hist_batch[0:len(hist), sampleId, k] = torch.from_numpy(hist[:, k])
                fut_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut[:, k])
                # fut_anchored_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut_anchored[:, k])
            op_mask_batch[0:len(fut), sampleId, :] = 1

            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            ve_matrix = torch.tensor(ve_matrix)
            ve_matrix_batch[sampleId, :] = ve_matrix
            ac_matrix = torch.tensor(ac_matrix)
            ac_matrix_batch[sampleId, :] = ac_matrix
            man_matrix = torch.tensor(man_matrix)
            man_matrix_batch[sampleId, :] = man_matrix
            view_mask = torch.tensor(view_mask)
            view_mask_batch[sampleId,:] =view_mask
            # Set up neighbor, neighbor sequence length, and mask batches:

            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    for k in range(self.ip_dim):
                        nbrs_batch[0:len(nbr), count, k] = torch.from_numpy(nbr[:, k])
                    pos = id % 11
                    mask_batch[sampleId, pos, :] = torch.ones(self.enc_size).byte()
                    count += 1
        max_num_object = 11

        graph_matrix = torch.zeros((256, max_num_object, 2))
        return hist_batch, nbrs_batch, nbr_list_len , mask_batch, fut_batch, lat_enc_batch, \
               lon_enc_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch,graph_matrix,view_mask_batch


# 0: Dataset Id
# 1: Vehicle Id
# 2: Frame Number
# 3: Local X
# 4: Local Y
# 5: Velocity (feet/s)
# 6: Acceleration (feet/s2)
# 7: Lane id
# 8: Lateral maneuver
# 9: Longitudinal maneuver
# 10-48: Neighbor Car Ids at grid location

