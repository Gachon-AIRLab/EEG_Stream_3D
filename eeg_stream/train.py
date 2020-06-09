import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import os
import sys
import shutil
import configparam
import time
import pickle
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('./model/')

from dataload import DataSplit
from dataload import eegDataset
from model.R2Plus1D import R2Plus1D
from model.C3D_tiny import c3d_tiny

def train(model, param):

    # assign parameters
    data_path = param.data_path
    num_subject = param.num_subject
    num_video = param.num_video
    learning_rate = param.learning_rate
    batch_size = param.batch_size
    num_epoch  = param.num_epoch



    # init data loader
    t0 = time.time()
    # def __init__(self, mode, data_path, num_sub, num_vid, num_seq, gt_type) (gt_type 1-valence / 2-arousal)
    data_set = eegDataset(0, data_path, num_subject, num_video, 60, 1)
    if param.use_predefined_idx:
        print('use pre-defined training list')
        with open(param.split_path + "train.txt", "rb") as fp:
            train_list = pickle.load(fp)
        with open(param.split_path + "test.txt", "rb") as fp:
            test_list = pickle.load(fp)
        sampler_train = SubsetRandomSampler(train_list)
        sampler_test = SubsetRandomSampler(test_list)
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=sampler_train, shuffle=False,
                                                  num_workers=12)
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=sampler_test, shuffle=False,
                                                   num_workers=12)
    else:
        split = DataSplit(data_set, param, True)
        train_loader, val_loader, test_loader = split.get_split(batch_size=batch_size, num_workers=12)


    t1 = time.time()
    print('dataset init: %f' % (t1 - t0))


    # prepare training
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_total = 0.0

    res_list_test = np.array([]).reshape((0, 3))

    for i in range(num_epoch):

        loss_epoch = 0.0
        cnt_epoch = 0

        t0 = time.time()

        idx = 0
        num_positive = 0
        num_total = 0
        for train_x, train_y in train_loader:
            #
            # idx = idx + 1
            # if idx > 5:
            #     break

            train_x = train_x.cuda()
            train_y = train_y.cuda()

            optimizer.zero_grad()
            output = model.forward(train_x)
            loss = loss_func(output, train_y)

            loss.backward()
            optimizer.step()

            output_sm = F.softmax(output, dim=1)
            _, output_index = torch.max(output_sm, 1)
            res = output_index.cpu().detach().numpy()

            tp = (res == train_y.cpu().detach().numpy()).sum()

            num_positive = num_positive + tp
            num_total = num_total + res.shape[0]

            loss_epoch = loss_epoch + loss
            cnt_epoch = cnt_epoch + 1

        scheduler.step()

        train_accuracy = num_positive / num_total

        idx = 0
        num_positive = 0
        num_total = 0



        loss_total = loss_total + (loss_epoch/cnt_epoch)


        for test_x, test_y in test_loader:

            # idx = idx + 1
            # if idx > 5:
            #     break

            test_x = test_x.cuda()
            test_y = test_y.cuda()

            with torch.no_grad():
                output = model.forward(test_x)

            output_sm = F.softmax(output, dim=1)
            _, output_index = torch.max(output_sm, 1)
            res = output_index.cpu().detach().numpy()

            tp = (res == test_y.cpu().detach().numpy()).sum()

            num_positive = num_positive + tp
            num_total = num_total + res.shape[0]

        test_accuracy = num_positive / num_total

        t1 = time.time()
        res_list_test = np.append(res_list_test, np.array([[i+1, train_accuracy, test_accuracy]]), axis=0)
        np.savetxt(param.result_path + 'evaluation_result.txt', res_list_test, fmt='%1.4f')

        print('epoch:{} loss:{} loss_avg:{} test_accuracy:{} time:{} lr:{}'.format(i+1, (loss_epoch/cnt_epoch), (loss_total/(i+1)), (num_positive/num_total), (t1-t0), scheduler.get_lr()))
        save_file_name = param.weight_path + param.weight_prefix + '_e{:04d}.pth'.format(i+1)
        torch.save(model.state_dict(), save_file_name)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
    else:
        conf_file_name = './config/train_c3d.cfg'
    print('loading configuration file: %s\n------'%conf_file_name)

    conf = configparam.ConfigParam()
    # conf.PrintConfig()
    conf.LoadConfiguration(conf_file_name)
    # conf.PrintConfig()

    # create directories
    if not os.path.exists(conf.result_path):
        os.makedirs(conf.result_path)
    if not os.path.exists(conf.weight_path):
        os.makedirs(conf.weight_path)
    if not os.path.exists(conf.split_path):
        os.makedirs(conf.split_path)
    print(conf_file_name)
    shutil.copy2(conf_file_name, conf.result_path + conf_file_name.split('/')[-1])
    print(conf.result_path + conf_file_name.split('/')[-1])

    t0 = time.time()

    if conf.train_method == 'c3d':
        print('c3d')
        model = c3d_tiny()
        shutil.copy2('./model/C3D_tiny.py', conf.result_path + './C3D_tiny.py')
    elif conf.train_method == 'r2plus1d':
        print('r2plus1d')
        model = R2Plus1D(2, (1, 1, 1, 1), 1)

    try:
        if conf.use_pretrained:
            model.load_state_dict(torch.load(conf.pretrained_name))
            print("model restored")
        else:
            print("learning from scratch")
        model.train()
        model.cuda()
    except:
        exit()
    t1 = time.time()
    print('model init: %f' % (t1 - t0))

    train(model, conf)
