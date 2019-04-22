import os, sys
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from face_alignment.utils import *
import face_alignment.models  as md


ops_list = [];

def conv_block():
    ConvBlock = nn.Sequential(
            nn.Conv2d(3,64,1),
            nn.ReLU(),
            nn.Conv2d(64,128,1),
            nn.Conv2d(128,128,1))
    return ConvBlock

class ConvBlockDown(torch.jit.ScriptModule):
    def __init__(self, in_plane, out_plane, down_size):
        super(ConvBlockDown, self).__init__()
        self.bn1 = BatchNorm2d(in_plane, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = Conv2d(in_plane, int(out_plane/2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(int(out_plane/2), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(int(out_plane/2), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(int(out_plane/4), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(int(out_plane/4), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.downsample = torch.jit.trace(nn.Sequential(
                    nn.BatchNorm2d(in_plane),
                    nn.ReLU(True),
                    nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=1, bias=False)
                ), torch.rand(1, in_plane, down_size, down_size))

    @torch.jit.script_method
    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out = torch.cat((out1,out2,out3),1)
        residual = self.downsample(residual)
        out += residual
        return out

class ConvBlockDown2(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ConvBlockDown2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.conv1 = Conv2d(in_plane, int(out_plane/2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_plane/2))
        self.conv2 = Conv2d(int(out_plane/2), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_plane/4))
        self.conv3 = Conv2d(int(out_plane/4), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.downsample = nn.Sequential(
                    nn.BatchNorm2d(in_plane),
                    nn.ReLU(True),
                    nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=1, bias=False)
                )

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

        out = torch.cat((out1,out2,out3),1)
        ops_list.append("convblock_cat")
        residual = self.downsample(residual)
        ops_list.append("convblock_downsample")
        out += residual
        ops_list.append("convblock_residual_add")
        return out

class ConvBlock(torch.jit.ScriptModule):
    def __init__(self, in_plane, out_plane):
        super(ConvBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_plane, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = Conv2d(in_plane, int(out_plane/2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(int(out_plane/2), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(int(out_plane/2), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(int(out_plane/4), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(int(out_plane/4), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    @torch.jit.script_method
    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out = torch.cat((out1,out2,out3),1)
        return out

class ConvBlock2(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ConvBlock2, self).__init__()
        self.bn1 = BatchNorm2d(in_plane)
        self.conv1 = Conv2d(in_plane, int(out_plane/2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(int(out_plane/2))
        self.conv2 = Conv2d(int(out_plane/2), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(int(out_plane/4))
        self.conv3 = Conv2d(int(out_plane/4), int(out_plane/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        ops_list.append('convblock_bn1')
        out1 = F.relu(out1, True)
        ops_list.append('convblock_bn1_relu')
        out1 = self.conv1(out1)
        ops_list.append('convblock_conv1')

        out2 = self.bn2(out1)
        ops_list.append('convblock_bn2')
        out2 = F.relu(out2, True)
        ops_list.append('convblock_bn2_relu')
        out2 = self.conv2(out2)
        ops_list.append('convblock_conv2')

        out3 = self.bn3(out2)
        ops_list.append('convblock_bn3')
        out3 = F.relu(out3, True)
        ops_list.append('convblock_bn3_relu')
        out3 = self.conv3(out3)
        ops_list.append('convblock_conv3')

        out = torch.cat((out1, out2, out3), 1)
        ops_list.append("convblock_cat")
        out += residual
        ops_list.append("convblock_residual_add")
        return out


class HourGlass2(nn.Module):
    def __init__(self):
        super(HourGlass2, self).__init__()
        self.b1_4 = ConvBlock2(256,256)
        self.b2_4 = ConvBlock2(256,256)
        self.b1_3 = ConvBlock2(256,256)
        self.b2_3 = ConvBlock2(256,256)
        self.b1_2 = ConvBlock2(256,256)
        self.b2_2 = ConvBlock2(256,256)
        self.b1_1 = ConvBlock2(256,256)
        self.b2_1 = ConvBlock2(256,256)
        self.b2_plus_1 = ConvBlock2(256,256)
        self.b3_1 = ConvBlock2(256,256)
        self.b3_2 = ConvBlock2(256,256)
        self.b3_3 = ConvBlock2(256,256)
        self.b3_4 = ConvBlock2(256,256)

    def forward(self, x):
        # Re4
        up4 = self.b1_4(x)
        ops_list.append("b1_4")
        inp4 = self.b2_4(F.avg_pool2d(x, 2, stride=2))
        ops_list.append("b2_4")

        # Re3
        up3 = self.b1_3(inp4)
        ops_list.append("b1_3")
        inp3 = self.b2_3(F.avg_pool2d(inp4, 2, stride=2))
        ops_list.append("b2_3")

        # Re2
        up2 = self.b1_2(inp3)
        ops_list.append("b1_2")
        inp2 = self.b2_2(F.avg_pool2d(inp3, 2, stride=2))
        ops_list.append("b2_2")

        # Re1
        up1 = self.b1_1(inp2)
        ops_list.append("b1_1")
        inp1 = self.b2_1(F.avg_pool2d(inp2, 2, stride=2))
        ops_list.append("b2_1")

        # Re1
        low2 = self.b2_plus_1(inp1)
        ops_list.append("b2_plus_1")
        low3 = self.b3_1(low2)
        ops_list.append("b3_1")
        out1 = up1 + F.upsample(low3, scale_factor=2, mode='nearest')
        ops_list.append("upsample")

        # Re2
        low3 = self.b3_2(out1)
        ops_list.append('b3_2')
        out2 = up2 + F.upsample(low3, scale_factor=2, mode='nearest')
        ops_list.append("upsample")

        # Re3
        low3 = self.b3_3(out2)
        ops_list.append('b3_3')
        out3 = up3 + F.upsample(low3, scale_factor=2, mode='nearest')
        ops_list.append('upsample')

        # Re4
        low3 = self.b3_4(out3)
        ops_list.append('b3_4')
        out4 = up4 + F.upsample(low3, scale_factor=2, mode='nearest')
        ops_list.append('upsample')

        return out4


class HourGlass(torch.jit.ScriptModule):
    __constants__ = ['num_modules','depth','features']

    def __init__(self, num_modules, depth, num_features, ch, h, w):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self.b1_4 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b2_4 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b1_3 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b2_3 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b1_2 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b2_2 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b1_1 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b2_1 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))

        self.b2_plus_1 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))

        self.b3_1 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b3_2 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b3_3 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))
        self.b3_4 = torch.jit.trace(ConvBlock(256, 256), torch.rand(1, ch, h, w))

    @torch.jit.script_method
    def forward(self, x):
        # Re4
        up4 = self.b1_4(x)
        inp4 = self.b2_4(F.avg_pool2d(x, 2, stride=2))

        # Re3
        up3 = self.b1_3(inp4)
        inp3 = self.b2_3(F.avg_pool2d(inp4, 2, stride=2))

        # Re2
        up2 = self.b1_2(inp3)
        inp2 = self.b2_2(F.avg_pool2d(inp3, 2, stride=2))

        # Re1
        up1 = self.b1_1(inp2)
        inp1 = self.b2_1(F.avg_pool2d(inp2, 2, stride=2))

        # Re1
        low2 = self.b2_plus_1(inp1)
        low3 = self.b3_1(low2)
        out1 = inp2 + F.upsample(low3, scale_factor=2, mode='nearest')

        # Re2
        low3 = self.b3_2(out1)
        out2 = inp3 + F.upsample(low3, scale_factor=2, mode='nearest')

        # Re3
        low3 = self.b3_3(out2)
        out3 = inp4 + F.upsample(low3, scale_factor=2, mode='nearest')

        # Re4
        low3 = self.b3_4(out3)
        out4 = x + F.upsample(low3, scale_factor=2, mode='nearest')

        return out4

class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlockDown2(64, 128)
        self.conv3 = ConvBlock2(128, 128)
        self.conv4 = ConvBlockDown2(128, 256)
        # level 1
        self.m0 = HourGlass2()
        self.top_m_0 = ConvBlock2(256, 256)
        self.conv_last0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end0 = nn.BatchNorm2d(256)
        self.l0 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al0 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 2
        self.m1 = HourGlass2()
        self.top_m_1 = ConvBlock2(256, 256)
        self.conv_last1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end1 = nn.BatchNorm2d(256)
        self.l1 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al1 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 3
        self.m2 = HourGlass2()
        self.top_m_2 = ConvBlock2(256,256)
        self.conv_last2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end2 = nn.BatchNorm2d(256)
        self.l2 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al2 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 4
        self.m3 = HourGlass2()
        self.top_m_3 = ConvBlock2(256, 256)
        self.conv_last3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end3 = nn.BatchNorm2d(256)
        self.l3 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        ops_list.append('conv1')
        x = self.conv2(x)
        ops_list.append('conv2')
        x = F.avg_pool2d(x, 2, stride=2)
        ops_list.append('conv2pool')
        x = self.conv3(x)
        ops_list.append('conv3')
        x = self.conv4(x)
        ops_list.append('conv4')

        # level 1
        hg = self.m0(x)
        ops_list.append("m0")
        ll = self.top_m_0(hg)
        ops_list.append("top_m_0")
        ll = F.relu(self.bn_end0(self.conv_last0(ll)), True)
        ops_list.append("relu->bn_end0->conv_last0")

        tmp_out = self.l0(ll)
        ops_list.append("l0")

        ll = self.bl0(ll)
        ops_list.append("bl0")
        tmp_out_ = self.al0(tmp_out)
        ops_list.append("al0")
        previous = x + ll + tmp_out_
        ops_list.append("Add_previous")

        # level 2
        hg = self.m1(previous)
        ops_list.append("m1")
        ll = self.top_m_1(hg)
        ops_list.append("top_m_1")
        ll = F.relu(self.bn_end1(self.conv_last1(ll)), True)
        ops_list.append("relu->bn_end1->conv_last1")

        tmp_out = self.l1(ll)
        ops_list.append("l1")

        ll = self.bl1(ll)
        ops_list.append("bl1")
        tmp_out_ = self.al1(tmp_out)
        ops_list.append("al1")
        previous = previous + ll + tmp_out_
        ops_list.append("Add_previous")

        # level 3
        hg = self.m2(previous)
        ops_list.append("m2")
        ll = self.top_m_2(hg)
        ops_list.append("top_m_2")
        ll = F.relu(self.bn_end2(self.conv_last2(ll)), True)
        ops_list.append("relu->bn_end2->conv_last2")

        tmp_out = self.l2(ll)
        ops_list.append("l2")

        ll = self.bl2(ll)
        ops_list.append("bl2")
        tmp_out_ = self.al2(tmp_out)
        ops_list.append("al2")
        previous = previous + ll + tmp_out_
        ops_list.append("Add_previous")

        # level 4
        hg = self.m3(previous)
        ops_list.append("m3")
        ll = self.top_m_3(hg)
        ops_list.append("top_m_3")
        ll = F.relu(self.bn_end3(self.conv_last3(ll)), True)
        ops_list.append("relu->bn_end3->conv_last3")
        tmp_out = self.l3(ll)
        ops_list.append("l3")
        outputs = tmp_out

        return outputs

class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlockDown2(64, 128)
        self.conv3 = ConvBlock2(128, 128)
        self.conv4 = ConvBlockDown2(128, 256)
        self.m0 = HourGlass2()
        self.top_m_0 = ConvBlock2(256, 256)
        self.conv_last0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end0 = nn.BatchNorm2d(256)
        self.l0 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al0 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 2
        self.m1 = HourGlass2()
        self.top_m_1 = ConvBlock2(256, 256)
        self.conv_last1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end1 = nn.BatchNorm2d(256)
        self.l1 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al1 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 3
        self.m2 = HourGlass2()
        self.top_m_2 = ConvBlock2(256,256)
        self.conv_last2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end2 = nn.BatchNorm2d(256)
        self.l2 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        self.bl2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.al2 = nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0)
        # level 4
        self.m3 = HourGlass2()
        self.top_m_3 = ConvBlock2(256, 256)
        self.conv_last3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn_end3 = nn.BatchNorm2d(256)
        self.l3 = nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        hg = self.m0(x)

        return hg 

ops_list2 = []
with open('ops_list2.txt','r') as f:
    for txt in f:
        ops_list2.append(txt)

ops_list1 = []
with open('ops_list.txt','r') as f:
    for txt in f:
        ops_list1.append(txt)


for i in range(len(ops_list2)):
    if ops_list2[i] != ops_list1[i]:
        print(ops_list2[i], '- ', ops_list1[i])

data_path = 'C:/Users/julius surya/AppData/Local/face_alignment/data/2DFAN-4.pth.tar'
pretrained_weights = torch.load(data_path, map_location="cpu")

inputs = torch.ones(1,3,256,256)

fakenet = Network2()
fakenet.load_state_dict(pretrained_weights, strict=False)
fakenet.eval()
fakemod = torch.jit.trace(fakenet, torch.rand(1,3,256,256))

out_fake = fakemod(inputs)

realnet = md.FAN(4)
realnet.load_state_dict(pretrained_weights, strict=False)
realnet.eval()

out_real = realnet(inputs)[-1]

print(torch.allclose(out_real, out_fake))







input ("........ final test here ...............")



net = Network2()
net.eval()

outs = net(torch.ones(2,3,256,256))
pts, pts_img = get_preds_fromhm(outs, torch.ones(2), 0.3)


net.load_state_dict(pretrained_weights, strict=True)
net.eval()

outs = net(torch.ones(2,3,256,256))
pts, pts_img = get_preds_fromhm(outs, torch.ones(2), 0.3)

mod = torch.jit.trace(net, torch.rand(1,3,256,256))
outs_mod = mod(torch.ones(1,3,256,256))

pts, pts_img = get_preds_fromhm(outs_mod, torch.ones(2), 0.3)



net_ori = md.FAN(4)
net_ori.load_state_dict(pretrained_weights, strict=True)
net_ori.eval()

outs = net_ori(torch.ones(1,3,256,256))[-1]

pts, pts_img = get_preds_fromhm(outs, torch.ones(2), 0.3)

#script_module = torch.jit.trace(Network2(), torch.rand(1,3,256,256))
#print(script_module)
#script_module.load_state_dict(pretrained_weights, strict=False)
#script_module.eval()
if torch.allclose(outs, outs_mod):
    print("Model is same,  now saving.....")
    mod.save('jit_test.pt')


"""
data_path = 'C:/Users/julius surya/AppData/Local/face_alignment/data/2DFAN-4.pth.tar'
weights = torch.load(data_path, map_location='cpu')
old_dict = []
for k, v in weights.items():
    old_dict.append(k)
script_module = Network()
script_module.eval()

#torch.save(script_module.state_dict(),'test_save.pt')
print(script_module)
model_dict = script_module.state_dict()
new_dict = []
for k, v in model_dict.items():
    if 'training' in k or 'num_batches_tracked' in k:
        continue
    else:
        new_dict.append(k)
#script_module.load_state_dict(weights)

for i in range(len(new_dict)):
    del model_dict[new_dict[i]]

mod_w = {}
for i in range(len(new_dict)):
    k = old_dict[i]
    v = weights[k]
    mod_w[k] = v

script_module.load_state_dict(mod_w, strict=False)
module = torch.jit.trace(script_module, torch.rand(1,3,256,256))
print("Inferencing")
outs = module(torch.rand(1,3,256,256))
print("Finished Inferencing")
module.save('jit_test.pt')
"""
