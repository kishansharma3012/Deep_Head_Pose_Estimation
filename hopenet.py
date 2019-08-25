import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class Hopenet_FeatStack(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Feature stack from 3 blocks of ResNet-50
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet_FeatStack, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.conv_feat1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn_feat1 = nn.BatchNorm2d(128)
        self.relu_feat1 = nn.ReLU(inplace=True)
        self.avgpool_feat1 = nn.AvgPool2d(56)        
        self.feat1 = nn.Sequential(self.conv_feat1, self.bn_feat1, self.relu_feat1, self.avgpool_feat1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv_feat2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn_feat2 = nn.BatchNorm2d(128)
        self.relu_feat2 = nn.ReLU(inplace=True)
        self.avgpool_feat2 = nn.AvgPool2d(28)
        self.feat2 = nn.Sequential(self.conv_feat2, self.bn_feat2, self.relu_feat2, self.avgpool_feat2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv_feat3 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, bias=False)
        self.bn_feat3 = nn.BatchNorm2d(128)
        self.relu_feat3 = nn.ReLU(inplace=True)
        self.avgpool_feat3 = nn.AvgPool2d(14)        
        self.feat3 = nn.Sequential(self.conv_feat3, self.bn_feat3, self.relu_feat3, self.avgpool_feat3)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.fc_yaw = nn.Linear(2048 + 128*3, num_bins)
        self.fc_pitch = nn.Linear(2048 + 128*3, num_bins)
        self.fc_roll = nn.Linear(2048 + 128*3, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_feat1 = self.feat1(x)
        x_feat1 = x_feat1.view(x_feat1.size(0), -1)
        
        x = self.layer2(x)
        x_feat2 = self.feat2(x)
        x_feat2 = x_feat2.view(x_feat2.size(0), -1)
        
        x = self.layer3(x)
        x_feat3 = self.feat3(x)
        x_feat3 = x_feat3.view(x_feat3.size(0), -1)
        
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = torch.cat([x, x_feat1, x_feat2, x_feat3], dim=1)

        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll
