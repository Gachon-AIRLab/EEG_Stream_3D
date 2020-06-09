import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import sys
import configparam
import time
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

sys.path.append('./model/')

from dataload import eegDataset
from eeg_flow_c3d_2c import EEG_Flow_C3D_2C

def eval(model, param):

    # assign parameters
    data_path = param.data_path
    num_subject = param.num_subject
    num_video = param.num_video
    batch_size = param.batch_size
    exception_subject = param.exception_subject

    # init data loader
    t0 = time.time()
    # def __init__(self, data_path, mode, num_sub, num_vid, num_seq, gt_type) (gt_type 1-valence / 2-arousal)
    data_set = eegDataset(1, data_path, num_subject, num_video, 60, 1)
    with open(param.split_path + "test.txt", "rb") as fp:
        test = pickle.load(fp)
    sampler_test = SubsetRandomSampler(test)
    test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=sampler_test, shuffle=False,
                                              num_workers=4)
    t1 = time.time()
    print('dataset init: %f' % (t1 - t0))


    num_total = 0
    num_positive = 0

    idx = 0

    for test_x, test_y in test_loader:

        idx = idx + 1
        if idx > 5:
            break

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


    print('accuracy %f  (%d / %d)'%(num_positive/num_total, num_positive, num_total))


if __name__ == '__main__':


    conf = configparam.ConfigParam()
    conf.LoadConfiguration('./config/test_sample.cfg')
    conf.PrintConfig()

    t0 = time.time()
    model = EEG_Flow_C3D_2C()
    try:
        if conf.use_pretrained:
            print(conf.pretrained_name)
            model.load_state_dict(torch.load(conf.pretrained_name))
            print("model restored")
        else:
            print("learning from scratch")

        model.eval()

        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
        #         m.track_running_stats = False

        model.cuda()
    except:
        exit()
    t1 = time.time()
    print('model init: %f' % (t1 - t0))

    eval(model, conf)