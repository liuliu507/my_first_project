import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

import math

class pyCNN(nn.Module):
    def __init__(self,Classes,FM=64,NC=30,para_tune=True):
        super(pyCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = NC,out_channels = FM,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 1, 1, ),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )


        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

    def forward(self, x1, x2):

        if x1.dim() != 4 or x1.size(1) != 30:
            print(f"Warning: HSI shape mismatch! Got {x1.shape}, expected [batch,30,h,w]")
        if x2.dim() != 4 or x2.size(1) != 1:
            print(f"Warning: LiDAR shape mismatch! Got {x2.shape}, expected [batch,1,h,w]")
            x2 = x2.unsqueeze(1)  # 自动添加通道维度

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)

        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        x = x1 + x2
        # x = x.view(x.size(0), 1, 2, -1)
        out3 = self.out3(x)

        # x = torch.cat([x1, x2], dim=1)
        # x = x.view(x.size(0), 1, 2, -1)
        # out3 = self.out3(x)
        return out1, out2, out3
