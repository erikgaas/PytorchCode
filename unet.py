import torch
from torch import nn
import pretrainedmodels
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

pretrained_settings = {
    'dpn68': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn68b': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-84854c156.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn92': {
        # 'imagenet': {
        #     'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224],
        #     'input_range': [0, 1],
        #     'mean': [124 / 255, 117 / 255, 104 / 255],
        #     'std': [1 / (.0167 * 255)] * 3,
        #     'num_classes': 1000
        # },
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn98': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn98-5b90dec4d.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn131': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn131-71dfe43e0.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn107': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-1ac7121e2.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    }
}

def dpnunet68(num_classes=1, pretrained='imagenet'):
    model = UnetDPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn68'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpnunet68b(num_classes=1, pretrained='imagenet+5k'):
    model = UnetDPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn68b'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpnunet92(num_classes=1, pretrained='imagenet+5k'):
    model = UnetDPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn92'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpnunet98(num_classes=1, pretrained='imagenet'):
    model = UnetDPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn98'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpnunet131(num_classes=1, pretrained='imagenet'):
    model = UnetDPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn131'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpnunet107(num_classes=1, pretrained='imagenet+5k'):
    model = UnetDPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn107'][pretrained]

        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):

    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense
    

class DecoderBlock(nn.Module):

    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        self.deccatbnact = CatBnAct(self.in_channels)

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, ptfeat, ufeat=None):
        if ufeat is not None:
            feat = ptfeat + (ufeat,)
            x = self.deccatbnact(feat)
        else:
            x = self.deccatbnact(ptfeat)

        return self.block(x)


class UnetDPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1, test_time_pool=False, num_filters=32):
        super(UnetDPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        self.dec5 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        self.ptfeat1 = nn.Sequential(blocks)
        self.dec4 = DecoderBlock(in_chs + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        blocks = OrderedDict()

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        self.ptfeat2 = nn.Sequential(blocks)
        self.dec3 = DecoderBlock(in_chs + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        blocks = OrderedDict()

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        self.ptfeat3 = nn.Sequential(blocks)
        self.dec2 = DecoderBlock(in_chs + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        blocks = OrderedDict()

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        #blocks['conv5_bn_ac'] = CatBnAct(in_chs)


        self.ptfeat4 = nn.Sequential(blocks)
        self.dec1 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 8)

        self.dec0 = DecoderBlock(in_chs, num_filters * 8 * 2, num_filters * 8)
        self.pool = nn.MaxPool2d(2, 2)

        self.decfinal = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, input):
        pt1 = self.ptfeat1(input)
        pt2 = self.ptfeat2(pt1)
        pt3 = self.ptfeat3(pt2)
        pt4 = self.ptfeat4(pt3)
        pt4 = torch.cat(pt4, dim=1)

        center = self.dec0(self.pool(pt4))
        dec1 = self.dec1(center)
        dec2 = self.dec2(pt3, dec1)
        dec3 = self.dec3(pt2, dec2)
        dc4 = self.dec4(pt1, dec3)

        dc_native = self.dec5(dc4)
        final = self.decfinal(dc_native)

        return final