# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT



class unet_3D_multi(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_multi, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4]*4, filters[3]*4, is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3]*4, filters[2]*4, is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2]*4, filters[1]*4, is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1]*4, filters[0]*4, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0]*4, n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        con_conv1 = []
        con_conv2 = []
        con_conv3 = []
        con_conv4 = []
        con_center = []
        for i in range(inputs.shape[1]):
            conv1 = self.conv1(inputs[:,i,:,:,:].unsqueeze(1))
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            center = self.center(maxpool4)
            center = self.dropout1(center)
            con_conv1.append(conv1)
            con_conv2.append(conv2)
            con_conv3.append(conv3)
            con_conv4.append(conv4)
            con_center.append(center)
        conv1 = torch.cat((con_conv1[0],con_conv1[1],con_conv1[2],con_conv1[3]), dim=1)
        conv2 = torch.cat((con_conv2[0],con_conv2[1],con_conv2[2],con_conv2[3]),dim=1)
        conv3 = torch.cat((con_conv3[0],con_conv3[1],con_conv3[2],con_conv3[3]),dim=1)
        conv4 = torch.cat((con_conv4[0],con_conv4[1],con_conv4[2],con_conv4[3]),dim=1)
        center = torch.cat((con_center[0],con_center[1],con_center[2],con_center[3]),dim=1)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final
# class unet_3D_multi(nn.Module):
#
#     def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
#         super(unet_3D_multi, self).__init__()
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#
#         filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#
#         self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#
#         self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#
#         self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
#
#         self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#
#         # upsampling
#         self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
#         self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
#         self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
#         self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)
#
#         # final conv (without any concat)
#         self.final = nn.Conv3d(filters[0], n_classes, 1)
#
#         self.dropout1 = nn.Dropout(p=0.3)
#         self.dropout2 = nn.Dropout(p=0.3)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm3d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         # con_conv1 = []
#         # con_conv2 = []
#         # con_conv3 = []
#         # con_conv4 = []
#         # con_center = []
#         con_final = []
#         for i in range(inputs.shape[1]):
#             conv1 = self.conv1(inputs[:,i,:,:,:].unsqueeze(1))
#             maxpool1 = self.maxpool1(conv1)
#
#             conv2 = self.conv2(maxpool1)
#             maxpool2 = self.maxpool2(conv2)
#
#             conv3 = self.conv3(maxpool2)
#             maxpool3 = self.maxpool3(conv3)
#
#             conv4 = self.conv4(maxpool3)
#             maxpool4 = self.maxpool4(conv4)
#
#             center = self.center(maxpool4)
#             center = self.dropout1(center)
#         #     con_conv1.append(conv1)
#         #     con_conv2.append(conv2)
#         #     con_conv3.append(conv3)
#         #     con_conv4.append(conv4)
#         #     con_center.append(center)
#         # conv1 = torch.cat((con_conv1[0],con_conv1[1],con_conv1[2],con_conv1[3]), dim=1)
#         # conv2 = torch.cat((con_conv2[0],con_conv2[1],con_conv2[2],con_conv2[3]),dim=1)
#         # conv3 = torch.cat((con_conv3[0],con_conv3[1],con_conv3[2],con_conv3[3]),dim=1)
#         # conv4 = torch.cat((con_conv4[0],con_conv4[1],con_conv4[2],con_conv4[3]),dim=1)
#         # center = torch.cat((con_center[0],con_center[1],con_center[2],con_center[3]),dim=1)
#
#             up4 = self.up_concat4(conv4, center)
#             up3 = self.up_concat3(conv3, up4)
#             up2 = self.up_concat2(conv2, up3)
#             up1 = self.up_concat1(conv1, up2)
#             up1 = self.dropout2(up1)
#
#             final = self.final(up1)
#             con_final.append(final)
#         # return final
#         outputs = torch.cat((con_final[0],con_final[1],con_final[2],con_final[3]),dim=1)
#         return outputs

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
class unet_3D_multi_feature(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_multi_feature, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # con_conv1 = []
        # con_conv2 = []
        # con_conv3 = []
        # con_conv4 = []
        con_center = []
        con_final = []
        for i in range(inputs.shape[1]):
            conv1 = self.conv1(inputs[:,i,:,:,:].unsqueeze(1))
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            center = self.center(maxpool4)
            center = self.dropout1(center)
        #     con_conv1.append(conv1)
        #     con_conv2.append(conv2)
        #     con_conv3.append(conv3)
        #     con_conv4.append(conv4)
        #     con_center.append(center)
        # conv1 = torch.cat((con_conv1[0],con_conv1[1],con_conv1[2],con_conv1[3]), dim=1)
        # conv2 = torch.cat((con_conv2[0],con_conv2[1],con_conv2[2],con_conv2[3]),dim=1)
        # conv3 = torch.cat((con_conv3[0],con_conv3[1],con_conv3[2],con_conv3[3]),dim=1)
        # conv4 = torch.cat((con_conv4[0],con_conv4[1],con_conv4[2],con_conv4[3]),dim=1)
        # center = torch.cat((con_center[0],con_center[1],con_center[2],con_center[3]),dim=1)

            up4 = self.up_concat4(conv4, center)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)
            up1 = self.dropout2(up1)

            final = self.final(up1)
            con_final.append(final)
        # return final
        outputs = torch.cat((con_final[0],con_final[1],con_final[2],con_final[3]),dim=1)
        return outputs,center

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
