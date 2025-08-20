import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from fadc import FADC
from FDConv_initialversion import FDConv
import math



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PyConv(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[1, 3, 5], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])

        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)

        return torch.cat((x1,x2,x3), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    return PyConv(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)



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
        # self.conv3 = nn.Sequential(
        #     get_pyconv(inplans=FM*2, planes=FM*4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
        #     nn.BatchNorm2d(FM*4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Dropout(0.5),
        #     )

        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        # FADC
        # self.conv3 = nn.Sequential(
        #     FADC(
        #         in_channels=FM * 2,
        #         out_channels=FM * 4,
        #         kernel_size=3,
        #         padding=1,
        #         dilation=1,
        #         offset_freq='FLC_high',
        #         fs_cfg={
        #             'k_list': [2, 4, 8],
        #             'lp_type': 'freq',
        #             'act': 'sigmoid',
        #         }
        #     ),
        #     nn.BatchNorm2d(FM * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Dropout(0.5),
        # )


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
        # self.conv6 = nn.Sequential(
        #     get_pyconv(inplans=FM * 2, planes=FM * 4, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]),
        #     nn.BatchNorm2d(FM * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Dropout(0.5),
        # )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        # FADC
        # self.conv6 = nn.Sequential(
        #     FADC(
        #         in_channels=FM * 2,
        #         out_channels=FM * 4,
        #         kernel_size=3,
        #         padding=1,
        #         dilation=1,
        #         offset_freq='FLC_high',
        #         fs_cfg={
        #             'k_list': [2, 4, 8],
        #             'lp_type': 'freq',
        #             'act': 'sigmoid',
        #         }
        #     ),
        #     nn.BatchNorm2d(FM * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Dropout(0.5),
        # )



        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        # self.out3 = nn.Linear(FM * 4, Classes)


        self.out3 = nn.Sequential(

            FADC(
                in_channels=FM * 4 * 2,  # 两个分支拼接后的通道数
                out_channels=FM * 4,
                kernel_size=3,
                padding=1,
                dilation=1,
                offset_freq='FLC_high',
                fs_cfg={
                    'k_list': [2, 4, 8],
                    'lp_type': 'freq',
                    'act': 'sigmoid',
                }
            ),

            # nn.Upsample(scale_factor=2),  # 上采样扩大尺寸
            # FDConv(
            #             in_channels=FM * 4 * 2,
            #             out_channels=FM * 4,
            #             kernel_size=(2, 1),
            #             padding=0,
            #             kernel_num=8,
            #             bias=True
            #         ),
            #
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(FM * 4, Classes)
        )


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

        h, w = x1.size(2), x1.size(3)

        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        # x = x1 + x2
        # # x = x.view(x.size(0), 1, 2, -1)
        # out3 = self.out3(x)

        x = torch.cat([x1, x2], dim=1)
        # x = x.view(x.size(0), 1, 2, -1)
        x = x.view(x.size(0), -1, h, w)
        out3 = self.out3(x)
        return out1, out2, out3


