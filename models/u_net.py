import torch
import torch.nn.functional as F
from torch import nn

from torchmeta import modules

from utils import initialize_weights

from collections import OrderedDict

class MetaConvTranspose2d(nn.ConvTranspose2d, modules.MetaModule):
    __doc__ = nn.ConvTranspose2d.__doc__

    def forward(self, input, output_size=None, params=None):
        
        if params is None:
            params = OrderedDict(self.named_parameters())
        weights = params.get('weight', None)
        bias = params.get('bias', None)
        
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, weights, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

class _MetaEncoderBlock(modules.MetaModule):

    def __init__(self, in_channels, out_channels, dropout=False):

        super(_MetaEncoderBlock, self).__init__()

        layers = [
            modules.MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout:

            layers.append(nn.Dropout())

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = modules.MetaSequential(*layers)

    def forward(self, x, params=None):

        return self.encode(x, self.get_subdict(params, 'encode'))

class _MetaDecoderBlock(modules.MetaModule):

    def __init__(self, in_channels, middle_channels, out_channels):

        super(_MetaDecoderBlock, self).__init__()

        self.decode = modules.MetaSequential(
            nn.Dropout2d(),
            modules.MetaConv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            MetaConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        )

    def forward(self, x, params=None):

        return self.decode(x, self.get_subdict(params, 'decode'))


class UNet(modules.MetaModule):

    def __init__(self, input_channels, num_classes, prototype=False):

        super(UNet, self).__init__()
        
        self.prototype = prototype

        self.enc1 = _MetaEncoderBlock(input_channels, 32)
        self.enc2 = _MetaEncoderBlock(32, 64)
        self.enc3 = _MetaEncoderBlock(64, 128, dropout=True)

        self.center = _MetaDecoderBlock(128, 256, 128)

        self.dec3 = _MetaDecoderBlock(256, 128, 64)
        self.dec2 = _MetaDecoderBlock(128, 64, 32)

        self.dec1 = modules.MetaSequential(
            nn.Dropout2d(),
            modules.MetaConv2d(64, 32, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(32),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(32, 32, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        if not self.prototype:
            self.final = modules.MetaConv2d(32, num_classes, kernel_size=1)
        
        initialize_weights(self)

    def forward(self, x, feat=False, params=None):
        
        enc1 = self.enc1(x, self.get_subdict(params, 'enc1'))
        enc2 = self.enc2(enc1, self.get_subdict(params, 'enc2'))
        enc3 = self.enc3(enc2, self.get_subdict(params, 'enc3'))
        
        center = self.center(enc3, self.get_subdict(params, 'center'))
        
        dec3 = self.dec3(torch.cat([center, F.interpolate(enc3, center.size()[2:], mode='bilinear')], 1), self.get_subdict(params, 'dec3'))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1), self.get_subdict(params, 'dec2'))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1), self.get_subdict(params, 'dec1'))
        
        if self.prototype: 
            return F.interpolate(dec1, x.size()[2:], mode='bilinear')
            
        else: 
            final = self.final(dec1, self.get_subdict(params, 'final'))

            if feat:
                return (F.interpolate(final, x.size()[2:], mode='bilinear'),
                        dec1,
                        F.interpolate(dec2, x.size()[2:], mode='bilinear'),
                        F.interpolate(dec3, x.size()[2:], mode='bilinear'),
                       )
            else:
                return F.interpolate(final, x.size()[2:], mode='bilinear')
