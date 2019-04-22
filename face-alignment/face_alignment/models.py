import torch
import torch.nn as nn
import torch.nn.functional as F
import math

ops_list = []

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        ops_list.append("convblock_bn1")
        out1 = F.relu(out1, True)
        ops_list.append("convblock_bn1_relu")
        out1 = self.conv1(out1)
        ops_list.append("convblock_conv1")

        out2 = self.bn2(out1)
        ops_list.append("convblock_bn2")
        out2 = F.relu(out2, True)
        ops_list.append("convblock_bn2_relu")
        out2 = self.conv2(out2)
        ops_list.append("convblock_conv2")

        out3 = self.bn3(out2)
        ops_list.append("convblock_bn3")
        out3 = F.relu(out3, True)
        ops_list.append("convblock_bn3_relu")
        out3 = self.conv3(out3)
        ops_list.append("convblock_conv3")

        out3 = torch.cat((out1, out2, out3), 1)
        ops_list.append("convblock_cat")

        if self.downsample is not None:
            residual = self.downsample(residual)
            ops_list.append("convblock_downsample")

        out3 += residual
        ops_list.append("convblock_residual_add")

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        print('b1_' + str(level))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        print('b2_' + str(level))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
            print('b2_plus_' + str(level))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))
        print('b3_' + str(level))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        ops_list.append(("b1_" + str(level)))

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        ops_list.append(("b2_" + str(level)))

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
            ops_list.append(("b2_plus_" + str(level)))

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        ops_list.append(("b3_" + str(level)))

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')
        ops_list.append(("Upsample" + str(level)))

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class FAN2(nn.Module):
    __constants__ = ['m0']
    def __init__(self, num_modules=1):
        super(FAN2, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        self.m0 = HourGlass(1, 4, 256)
        # Stacking part
        # for hg_module in range(self.num_modules):
        #     self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
        #     self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        #     self.add_module('conv_last' + str(hg_module),
        #                     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        #     self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
        #     self.add_module('l' + str(hg_module), nn.Conv2d(256,
        #                                                     68, kernel_size=1, stride=1, padding=0))

        #     if hg_module < self.num_modules - 1:
        #         self.add_module(
        #             'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        #         self.add_module('al' + str(hg_module), nn.Conv2d(68,
        #                                                          256, kernel_size=1, stride=1, padding=0))

    # @torch.jit.script_method
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        # previous = x
        hg = self._modules['m'+ str(0)](x)
        # for i in range(1):
            # hg = self._modules['m' + str(i)](previous)

        outputs = hg 
        return outputs

# class FAN(torch.jit.ScriptModule):
class FAN(nn.Module):
    __constants__ = ['m0']
    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))

    # @torch.jit.script_method
    def forward(self, x):
        print("input", x.size())
        x = F.relu(self.bn1(self.conv1(x)), True)
        ops_list.append("conv1")
        x = self.conv2(x)
        ops_list.append("conv2")
        x = F.avg_pool2d(x, 2, stride=2)
        ops_list.append("conv2pool")
        x = self.conv3(x)
        ops_list.append("conv3")
        x = self.conv4(x)
        ops_list.append("conv4")

        previous = x

        outputs = []
        # for i in range(self.num_modules):
        for i in range(4):
            hg = self._modules['m' + str(i)](previous)
            ops_list.append(("m"+str(i)))

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ops_list.append(("top_m_" + str(i)))

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)
            ops_list.append(("relu->bn_end" + str(i) + "->conv_last" + str(i)))

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            ops_list.append('l' + str(i))
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                ops_list.append('bl' + str(i))
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                ops_list.append('al' + str(i))
                previous = previous + ll + tmp_out_
                ops_list.append('Add_previous')

        return outputs


class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.fc(x)

        return x
