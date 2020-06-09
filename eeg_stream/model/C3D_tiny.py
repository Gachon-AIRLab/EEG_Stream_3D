
import torch
import torch.nn as nn

# 3D CNN
class c3d_tiny(nn.Module):
    def __init__(self):
        super(c3d_tiny, self).__init__()

        self.dbd = True

        # 1*1*128*41*41
        self.conv_1_64 = nn.Conv3d(1, 64, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv_64_128 = nn.Conv3d(64, 128, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv_128_256 = nn.Conv3d(128, 256, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv_256_256_1 = nn.Conv3d(256, 256, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv_256_256_2 = nn.Conv3d(256, 256, kernel_size=(7, 3, 3), padding=(3, 1, 1))

        self.maxpool_t1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.batch_norm3_64_1 = nn.BatchNorm3d(64)
        self.batch_norm3_128_1 = nn.BatchNorm3d(128)
        self.batch_norm3_256_1 = nn.BatchNorm3d(256)
        self.batch_norm3_256_2 = nn.BatchNorm3d(256)
        self.batch_norm3_256_3 = nn.BatchNorm3d(256)

        self.AdaptiveMaxPool3d = nn.AdaptiveMaxPool3d(output_size=(4, 2, 2))


        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # h = self.pool0(x)
        h = self.conv_1_64(x)
        h = self.batch_norm3_64_1(h)
        h = self.relu(h)
        h = self.maxpool_t1(h)

        h = self.conv_64_128(h)
        h = self.batch_norm3_128_1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.conv_128_256(h)
        h = self.batch_norm3_256_1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.conv_256_256_1(h)
        h = self.batch_norm3_256_2(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.conv_256_256_2(h)
        h = self.batch_norm3_256_3(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = h.view(h.size(0), -1)

        if self.dbd:
            print('flatten')
            print(h.size())
            self.dbd = False

        h = self.fc1(h)
        h = self.relu(h)
        h - self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h - self.dropout(h)
        h = self.fc3(h)


        probs = h

        if self.dbd:
            print('probs')
            print(probs.size())
            print(probs)

        return probs
