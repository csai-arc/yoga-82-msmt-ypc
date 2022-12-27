from __future__ import division

import torch
import torch.nn as nn
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels=None, activation=None, dilation=1, downsample=False, proj_ratio=4, 
                        upsample=False, asymetric=False, regularize=True, p_drop=None, use_prelu=True):
        super(BottleNeck, self).__init__()

        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        #if out_channels == 1:
        #   inplace = False
        #else:
        #   inplace = True
        if out_channels is None: out_channels = in_channels
        else: self.pad = out_channels - in_channels

        if regularize: assert p_drop is not None
        if downsample: assert not upsample
        elif upsample: assert not downsample
        inter_channels = in_channels//proj_ratio

        # Main
        #if upsample:
        #    self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #    self.bn_up = nn.BatchNorm2d(out_channels)
        #    self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        #elif downsample:
        #    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        if downsample: 
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=False)

        if asymetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1,5), padding=(0,2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU() if use_prelu else nn.ReLU(inplace=False),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(5,1), padding=(2,0)),
            )
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=False)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=False)

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=False)

    def forward(self, x, indices=None, output_size=None):
        # Main branch
        #identity = x
        #if self.upsample:
            #assert (indices is not None) and (output_size is not None)
        #identity = self.bn_up(self.spatil_conv(identity))
            #if identity.size() != indices.size():
            #    pad = (indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0)
            #    identity = F.pad(identity, pad, "constant", 0)
        #identity = self.unpool(identity)#, indices=indices)#, output_size=output_size)
        #elif self.downsample:
        #    identity, idx = self.pool(identity)

        '''
        if self.pad > 0:
            if self.pad % 2 == 0 : pad = (0, 0, 0, 0, self.pad//2, self.pad//2)
            else: pad = (0, 0, 0, 0, self.pad//2, self.pad//2+1)
            identity = F.pad(identity, pad, "constant", 0)
        '''

        #if self.pad > 0:
        #    extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
        #    if torch.cuda.is_available(): extras = extras.cuda(0)
        #    identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        #if identity.size() != x.size():
        #    pad = (identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0)
        #    x = F.pad(x, pad, "constant", 0)

        #x += identity
        x = self.prelu_out(x)

        if self.downsample:
            return x, idx
        return x
        
class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CFAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        inter_channels = 1024
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output


'''class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
'''
class EffNetV2yoga(nn.Module):
    def __init__(self, cfgs1,cfgs2,cfgs3,cfgs4,cfgs5,cfgs6, num_classes=1000, width_mult=1.):
        super(EffNetV2yoga, self).__init__()
        self.cfgs1 = cfgs1
        self.cfgs2 = cfgs2
        self.cfgs3 = cfgs3
        self.cfgs4 = cfgs4
        self.cfgs5 = cfgs5
        self.cfgs6 = cfgs6

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        #print("###########################################",input_channel)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs1:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        features_output_channel = output_channel
        #print("###########################################",output_channel)
        layers2 = []
        layers3 = []
        layers4 = []
        layers5 = []
        layers6 = []
        
        input_channel = features_output_channel
        for t, c, n, s, use_se in self.cfgs2:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers2.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        features2_output_channel = output_channel
        self.features2 = nn.Sequential(*layers2)
        
        input_channel = features_output_channel
        for t, c, n, s, use_se in self.cfgs3:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers3.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        features3_output_channel = output_channel
        self.features3 = nn.Sequential(*layers3)
        
        input_channel = (features2_output_channel+features3_output_channel)
        for t, c, n, s, use_se in self.cfgs4:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers4.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        features4_output_channel = output_channel
        self.features4 = nn.Sequential(*layers4)
        
        input_channel = features4_output_channel
        for t, c, n, s, use_se in self.cfgs5:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers5.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        features5_output_channel = output_channel
        self.features5 = nn.Sequential(*layers5)
        
        input_channel = features5_output_channel
        for t, c, n, s, use_se in self.cfgs6:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers6.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        features6_output_channel = output_channel
        self.features6 = nn.Sequential(*layers6)
        
        # building last several layers
        #output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.yoga6_conv = conv_1x1_bn(features4_output_channel, output_channel)
        self.yoga6_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.yoga6_classifier = nn.Linear(output_channel, 6)

        self.yoga20_conv = conv_1x1_bn(features5_output_channel, output_channel)
        self.yoga20_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.yoga20_classifier = nn.Linear(output_channel, 20)
        
        self.yoga82_conv = conv_1x1_bn(features6_output_channel, output_channel)
        self.yoga82_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.yoga82_classifier = nn.Linear(output_channel, 82)
        #print("reached end",output_channel)
        
        ################ Class activation maps network architecture #############################
        self.cfam = CFAMBlock(features4_output_channel+features5_output_channel+features6_output_channel, 1024)
        
        ##########Decoder################
        self.bottleneck1 = BottleNeck(1024, 512, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck2 = BottleNeck(512, 256, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck3 = BottleNeck(256, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck4 = BottleNeck(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck5 = BottleNeck(16, 1, upsample=True, p_drop=0.1, use_prelu=False)  # change output size to 1 after including layer-cam loss
        self.bottleneck6 = BottleNeck(16, 1, upsample=True, p_drop=0.1, use_prelu=False)  # change output size to 1 after including layer-cam loss

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x2 = self.features2(x)
        x3 = self.features3(x)
        xcat = torch.cat((x2, x3), dim=1)
        x4 = self.features4(xcat)
        x5 = self.features5(x4)
        x6 = self.features6(x5)
        
        
        yoga6_x = self.yoga6_conv(x4)
        yoga6_x = self.yoga6_avgpool(yoga6_x)
        yoga6_x = yoga6_x.view(yoga6_x.size(0), -1)
        yoga6_x = self.yoga6_classifier(yoga6_x)

        yoga20_x = self.yoga20_conv(x5)
        yoga20_x = self.yoga20_avgpool(yoga20_x)
        yoga20_x = yoga20_x.view(yoga20_x.size(0), -1)
        yoga20_x = self.yoga20_classifier(yoga20_x)

        yoga82_x = self.yoga82_conv(x6)
        yoga82_x = self.yoga82_avgpool(yoga82_x)
        yoga82_x = yoga82_x.view(yoga82_x.size(0), -1)
        yoga82_x = self.yoga82_classifier(yoga82_x)
        
        ################ Class activation maps network architecture ####################
        ycat_hm = torch.cat((x4,x5,x6), dim=1)
        y = self.cfam(ycat_hm)
        
        #decoder
        y = self.bottleneck1(y)
        y = self.bottleneck2(y)
        y = self.bottleneck3(y)
        y1 = self.bottleneck4(y)
        y2 = self.bottleneck5(y1)
        y3 = self.bottleneck6(y1)
        



        return yoga6_x, yoga20_x, yoga82_x, y2, y3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
                
def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs1 = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
    ]
    cfgs2 = [
        [6, 256, 15, 2, 1],
    ]
    cfgs3 = [
        [6, 160, 2, 1, 1],
        [6, 256, 2, 2, 1],
    ]
    cfgs4 = [
        [2, 512, 4, 1, 1],
    ]
    cfgs5 = [
        [2, 512, 4, 1, 1],
    ]
    cfgs6 = [
        [2, 512, 4, 1, 1],
    ]
    return EffNetV2yoga(cfgs1,cfgs2,cfgs3,cfgs4,cfgs5,cfgs6, **kwargs)


#def effnetv2_s(**kwargs):
#    """
#    Constructs a EfficientNetV2-S model
#    """
#    cfgs = [
#        # t, c, n, s, SE
#        [1,  24,  2, 1, 0],
#        [4,  48,  4, 2, 0],
#        [4,  64,  4, 2, 0],
#        [4, 128,  6, 2, 1],
#        [6, 160,  9, 1, 1],
#        [6, 256, 15, 2, 1],
#    ]
#    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
