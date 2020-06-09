import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
import cv2

# get average and std per each subject


base_dir = './data_preprocessed_python/'
target_dir_z_float = '../dataset/deap_eeg_flow/'

LUT = np.load('eeg_pos.npy').astype(np.uint32)

print(LUT)
exit()

num_sub = 32
num_video = 40
num_ch = 32

grid_x = 9
grid_y = 9
grid_up = 64

# video sequence 파일
rr, cc = np.mgrid[0:grid_y, 0:grid_x]
grid = np.zeros([grid_y, grid_x], dtype=np.float32)

for s in range(num_sub):
    x = cPickle.load(open(base_dir + 's%02d.dat'%(s+1), 'rb'), encoding='latin1')
    data_all = x['data']
    labels_all = x['labels']

    data_all_sub = np.ndarray(shape=(32, 40 * 7680), dtype=np.float32)

    print('sub:' + str(s + 1) + ' processing..')
    for v in range(num_video):

        data_all_sub[:, v*7680:(v+1)*7680] = data_all[v, 0:32, 128*3:].squeeze()

    data_mean = np.mean(data_all_sub[:, :], axis=1).reshape(-1, 1)
    data_std = np.std(data_all_sub[:, :], axis=1).reshape(-1, 1)

    for v in range(num_video):

        print('sub:' + str(s+1) + ' vid:' + str(v+1))

        # shape: 32 X 8064
        data = data_all[v, 0:32, :]
        labels = labels_all[v]

        # check label(valence)
        valence = 0;
        if labels[0] >= 5:
            valence = 1

        arousal = 0;
        if labels[1] >= 5:
            arousal = 1

        chuck_z_u = np.ndarray(shape=(128, grid_up, grid_up), dtype=np.uint8)
        chunk_z_f = np.ndarray(shape=(128, grid_up, grid_up), dtype=np.float32)
        chuck_mm_u = np.ndarray(shape=(128, grid_up, grid_up), dtype=np.uint8)
        chunk_mm_f = np.ndarray(shape=(128, grid_up, grid_up), dtype=np.float32)
        chunk_idx = 0
        # 시작 후 3초는 버리므로, frame idx는 128*3 부터 시작함
        frame_idx = 128 * 3

        # for t in range(8064):
        # 3초 버리고, 128개 sample 취함
        for i in range(60):

            # chunk shape: 32 x 128
            chunk = data[:, frame_idx + (128 * i):frame_idx + (128 * (i+1))]

            # chunk_mm = (chunk - data_min) / (data_max - data_min)
            chunk_z = (chunk - data_mean) / data_std

            for t in range(128):

                grid[:] = np.nan
                grid[LUT[:,0], LUT[:,1]] = chunk_z[:, t]

                # Z
                vals = ~np.isnan(grid)
                interp_func = interpolate.Rbf(rr[vals], cc[vals], grid[vals], function='gaussian')
                grid_norm = interp_func(rr, cc)
                grid_norm_up = cv2.resize(grid_norm, (grid_up, grid_up), interpolation = cv2.INTER_CUBIC)
                chunk_z_f[t, :, :] = grid_norm_up

            target_path_eflow_z = '{}s{:02d}_v{:02d}_t{:02d}_stream.npy'.format(target_dir_z_float, s + 1, v + 1, i+1)
            target_path_valence_z = '{}s{:02d}_v{:02d}_t{:02d}_valence.txt'.format(target_dir_z_float, s + 1, v + 1, i + 1)
            target_path_arousal_z = '{}s{:02d}_v{:02d}_t{:02d}_arousal.txt'.format(target_dir_z_float, s + 1, v + 1, i + 1)

            np.save(target_path_eflow_z, chunk_z_f)
            file_valence = open(target_path_valence_z, 'w')
            file_valence.write(str(valence))
            file_valence.close()
            file_arousal = open(target_path_arousal_z, 'w')
            file_arousal.write(str(arousal))
            file_arousal.close()
