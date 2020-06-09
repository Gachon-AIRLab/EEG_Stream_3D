import torch
import pickle
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import configparam

""" 
torch.utils.data.DataLoader 옵션

DataLoader(dataset=dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
"""
class DataSplit:
    def __init__(self, dataset, param, shuffle=False):
    # def __init__(self, dataset, save_path, exception_list, shuffle=False):
        self.dataset = dataset
        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))

        data_per_video = 30
        dataset_per_sub = 40 * data_per_video
        self.filtered_indices = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        if param.dataset_type == 'kfold':
            print('kfold kfold')
            if param.target_subject[0] == 0:
                param.target_subject = list(range(1, param.num_subject+1))
            if param.target_video[0] == 0:
                param.target_video = list(range(1, param.num_video+1))

            for i in range(len(param.target_subject)):
                s = param.target_subject[i] - 1
                for j in range(len(param.target_video)):
                    v = param.target_video[j] - 1
                    self.filtered_indices.extend(self.indices[s*dataset_per_sub + v*data_per_video:s*dataset_per_sub + v*data_per_video + data_per_video])
            if shuffle:
                np.random.shuffle(self.filtered_indices)
            num_train = int(np.floor(len(self.filtered_indices) * param.train_ratio))
            self.train_indices = self.filtered_indices[:num_train]
            self.test_indices = self.filtered_indices[num_train:]

        elif param.dataset_type == 'loo-s':
            print('loo-s loo-s')

            for i in range(param.num_subject):
                if i+1 in param.exception_subject:
                    self.test_indices.extend(self.indices[i*dataset_per_sub:i*dataset_per_sub+dataset_per_sub])
                else:
                    self.train_indices.extend(self.indices[i*dataset_per_sub:i*dataset_per_sub+dataset_per_sub])
            if shuffle:
                np.random.shuffle(self.train_indices)
                np.random.shuffle(self.test_indices)

        elif param.dataset_type == 'loo-v':
            print('loo-v loo-v')
            for i in range(len(param.target_subject)):
                s = param.target_subject[i] - 1
                for v in range(param.num_video):
                    if v+1 in param.exception_video:
                        self.test_indices.extend(self.indices[s*dataset_per_sub + v*60:s*dataset_per_sub + v*60 + 60])
                    else:
                        self.train_indices.extend(self.indices[s*dataset_per_sub + v*60:s*dataset_per_sub + v*60 + 60])
            if shuffle:
                np.random.shuffle(self.train_indices)
                np.random.shuffle(self.test_indices)

        print(self.test_indices)
        print(len(self.test_indices))

        with open(param.split_path + "train.txt", "wb") as fp:
            pickle.dump(self.train_indices, fp)
        # with open(param.split_path + "val.txt", "wb") as fp:
        #     pickle.dump(self.val_indices, fp)
        with open(param.split_path + "test.txt", "wb") as fp:
            pickle.dump(self.test_indices, fp)

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_split(self, batch_size=50, num_workers=4):
        print('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        print("DONE\n")
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        print('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        print('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_test_loader(self, batch_size=50, num_workers=4):
        print('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


# Custom dataloader 만드는 곳
class eegDataset(Dataset):

    # [필수] data read, download 하는 곳
    # mode: 0-train / 1-test
    # data_path: training dataset location
    # num_sub: number of subjects
    # num_vid: number of videos
    # num_seq: number of sequences
    # gt_type: 1-valence / 2-arousal
    def __init__(self, dmode, data_path, num_sub, num_vid, num_seq, target_label):
        self.mode = dmode
        # datapath 위치
        self.datapath = data_path
        # eeg data list
        self.eeg_data = []
        # eeg data G.T list
        self.eeg_GT = []

        if target_label == 'valence':
            print('----------------valence--------------')
            gt_type = 1
        elif target_label == 'arousal':
            print('----------------arousal--------------')
            gt_type = 2
        elif target_label == 'both':
            print('----------------both--------------')
            gt_type = 3


        # S, V, T 범위 변경
        # 현재 S = 1, v = (0 ~ 5), t = (0 ~ 60) 설정 = 300개
        print('--------------------- num seq: %d==='%num_seq)
        for s in range(num_sub):
            for v in range(num_vid):
                for t in range(num_seq):
                    self.eeg_data.append(self.datapath+"s%.2d_v%.2d_t%.2d_stream.npy"%(s+1,v+1,t+1))
                    if gt_type == 1:
                        self.eeg_GT.append(self.datapath + "s%.2d_v%.2d_t%.2d_valence.txt" % (s + 1, v + 1, t + 1))
                    elif gt_type == 2:
                        self.eeg_GT.append(self.datapath + "s%.2d_v%.2d_t%.2d_arousal.txt" % (s + 1, v + 1, t + 1))
                    elif gt_type == 3:
                        self.eeg_GT.append(self.datapath + "s%.2d_v%.2d_t%.2d_multi.txt" % (s + 1, v + 1, t + 1))

        self.len = len(self.eeg_data)
        print(self.len)

    # [필수] index에 해당하는 아이템을 넘겨주는 곳
    def __getitem__(self, index):
        x = np.load(self.eeg_data[index]).astype(np.float32)

        f = open(self.eeg_GT[index], 'r')
        y = int(f.read())
        f.close()

        # When training mode, add Gaussian noise
        if self.mode == 0:
            x = x + np.random.normal(0, 0.1, x.shape)

            # i = np.random.randint(2, size=1)
            # if i[0] == 1:
            #     x = np.flip(x, 0)

        x = x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])
        x = x.astype(np.float32)

        return x, y

    # [선택] data size를 넘겨주는 파트
    def __len__(self):
        return self.len

