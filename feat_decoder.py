
import torch.nn as nn
from torch.nn import functional as F 


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(self,hidden_planes, mode, conv_mask = False, use_se = False):
        super(Block, self).__init__()
        self.use_se = use_se
        if mode == 'down':
            self.conv = conv3x3(hidden_planes, int(hidden_planes*0.5), stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(int(hidden_planes*0.5),momentum=0.1)

        elif mode == 'up':
            self.conv = conv3x3(hidden_planes, hidden_planes*2, stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(hidden_planes*2,momentum=0.1)

        else:
            self.conv = conv3x3(hidden_planes, hidden_planes, stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(hidden_planes,momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class MultiScaleFeatFusion(nn.Module):
    
    def __init__(self, channel_dim_lst, expand_ratio=0.5):
        super().__init__()

        d1,d2,d3 = channel_dim_lst[0], channel_dim_lst[1], channel_dim_lst[2]
        
        self.relu = nn.ReLU()
        
        self.conv1_11 = conv3x3(d1, d2)
        self.conv2_11 = conv3x3(d2, d3)

        self.bn1 = nn.BatchNorm2d(d1,momentum=0.1)
        self.bn2 = nn.BatchNorm2d(d2,momentum=0.1)
        self.bn3 = nn.BatchNorm2d(d3,momentum=0.1)

        self.conv1_33 = conv3x3(d1, d1)
        self.conv2_33 = conv3x3(d2, d2)
        self.conv3_33 = conv3x3(d3, d3)

        self.dn_block1 = Block(2*d1, 'down', False, False)
        self.dn_block2 = Block(2*d2, 'down', False, False)
        self.dn_block3 = Block(2*d3, 'down', False, False)
        

    def forward(self, x1, x2, x3): 
        s2 = x2.shape[2]
        s3 = x3.shape[2]
        
        x1 = self.dn_block1(x1)
        x2 = self.dn_block2(x2)
        x3 = self.dn_block3(x3)
    
        x1 = self.conv1_33(x1) 
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x1_out = x1
        x1 = F.interpolate(x1, size=[s2, s2], mode='bilinear')
        x1 = self.conv1_11(x1)
        
        x2 = self.conv2_33(x2) 
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x1
        x2_out = x2
        
        x2 = F.interpolate(x2, size=[s3, s3], mode='bilinear')
        x2 = self.conv2_11(x2)
        
        x3 = self.conv3_33(x3) 
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x2 + x3

        return x1_out, x2_out, x3


class MultiScaleFeatFusionReverse(nn.Module):
    
    def __init__(self, channel_dim_lst, expand_ratio=0.5):
        super().__init__()

        d1,d2,d3 = channel_dim_lst[0], channel_dim_lst[1], channel_dim_lst[2]
        
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(d1,momentum=0.1)
        self.bn2 = nn.BatchNorm2d(d2,momentum=0.1)
        self.bn3 = nn.BatchNorm2d(d3,momentum=0.1)
    
        self.conv1_11 = conv3x3(d3, d2)
        self.conv2_11 = conv3x3(d2, d1)
        
        self.conv1_33 = conv3x3(d3, d3)
        self.conv2_33 = conv3x3(d2, d2)      
        self.conv3_33 = conv3x3(d1, d1)
        
        self.up_block1 = Block(d1, 'up', False, False)
        self.up_block2 = Block(d2, 'up', False, False)
        self.up_block3 = Block(d3, 'up', False, False)
        

    def forward(self, x1, x2, x3):
        s1 = x1.shape[2]
        s2 = x2.shape[2]
    
        x3 = self.conv1_33(x3) 
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3_out = x3
        x3 = F.interpolate(x3, size=[s2, s2], mode='bilinear')
        x3 = self.conv1_11(x3)
        
        x2 = self.conv2_33(x2) 
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x3
        x2_out = x2
        
        x2 = F.interpolate(x2, size=[s1, s1], mode='bilinear')
        x2 = self.conv2_11(x2)
        
        x1 = self.conv3_33(x1) 
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = x2 + x1
        
        x1 = self.up_block1(x1)
        x2_out = self.up_block2(x2_out)
        x3_out = self.up_block3(x3_out)

        return x1, x2_out, x3_out


class FeatEncoder(nn.Module):
    def __init__(self,planes):
        super(FeatEncoder, self).__init__()
        
        hidden_planes = int(planes*0.5)
        self.bn1 = nn.BatchNorm2d(hidden_planes, momentum=0.1)
        self.bn2 = nn.BatchNorm2d( int(hidden_planes*0.5), momentum=0.1)
        
        self.conv0 = conv1x1(planes, planes)
        self.conv1 = conv1x1(planes, hidden_planes)
        self.conv2 = conv1x1(hidden_planes, int(hidden_planes*0.5))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, out):
        out = self.conv0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        return out


class FeatDecoder(nn.Module):
    def __init__(self,planes):
        super(FeatDecoder, self).__init__()
        
        hidden_planes = int(planes*0.5)

        self.conv3 = conv1x1(planes, planes)
        
        self.conv1 = conv1x1(int(hidden_planes*0.5), hidden_planes)
        self.bn2 = nn.BatchNorm2d(hidden_planes,momentum=0.1)
        
        self.conv2 = conv1x1(hidden_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes,momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, out):
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv3(out)
 
        return out



class DecoderConcat(nn.Module):
    def __init__(self, device, channel_dim_lst):
        super(DecoderConcat, self).__init__()
        self.device = device

        self.e1 = FeatEncoder(channel_dim_lst[0])
        self.e2 = FeatEncoder(channel_dim_lst[1])
        self.e3 = FeatEncoder(channel_dim_lst[2])

        self.d1 = FeatDecoder(channel_dim_lst[0])
        self.d2 = FeatDecoder(channel_dim_lst[1])
        self.d3 = FeatDecoder(channel_dim_lst[2])
        
        self.msff  = MultiScaleFeatFusion([int(i * 0.125) for i in channel_dim_lst])
        self.msffr = MultiScaleFeatFusionReverse([int(i * 0.125) for i in channel_dim_lst])
        
        
    def forward(self,x_lst):

        x1,x2,x3 = x_lst[0], x_lst[1], x_lst[2]
        
        x1 = self.e1(x1)
        x2 = self.e2(x2)
        x3 = self.e3(x3)
    
        x1,x2,x3 = self.msff(x1,x2,x3)
        x1,x2,x3 = self.msffr(x1,x2,x3)
        
        x1 = self.d1(x1)
        x2 = self.d2(x2)
        x3 = self.d3(x3)

        return x1,x2,x3


