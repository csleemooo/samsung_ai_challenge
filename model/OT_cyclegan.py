import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, args, input_channel=1):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.output_channel = 1   # check ?!
        self.use_norm = args.disc_norm
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode
        self.patch_gan=args.patch_gan
        self.lrelu_slope = 0.1
        self.disc_classifier = args.disc_classifier
        self.args = args

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2

        self.m = []
        c = args.initial_channel
        self.m += [CBR(in_channel=3, out_channel=c, use_norm=False, kernel=4, padding=1, stride=2,
                       lrelu_use=self.lrelu_use, slope=self.lrelu_slope)]
        for r in range(3):
            self.m += [CBR(in_channel=c, out_channel=c*2, use_norm=False, kernel=4, padding=1, stride=2,
                       lrelu_use=self.lrelu_use, slope=self.lrelu_slope)]
            c*= 2

        self.m = nn.Sequential(*self.m)

        if self.patch_gan:
            self.conv_out = nn.Conv2d(in_channels=c4, out_channels=self.output_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            k_size = int(args.img_size/np.power(2, 4))
            self.conv_out = nn.Conv2d(in_channels=c4, out_channels=1, kernel_size=(k_size, k_size), bias=False)

        if self.disc_classifier:
            k_size = c5
            k_size = int(args.img_size/np.power(2, 4))
            if args.distance_regression:
                self.conv_out_cls = nn.Conv2d(in_channels=c4, out_channels=1, kernel_size=(k_size, k_size), bias=False)
            else:
                self.conv_out_cls = nn.Conv2d(in_channels=c4, out_channels=args.N_classes, kernel_size=(k_size, k_size), bias=False)

        self.apply(weights_initialize_xavier_normal)

    def forward(self, x, out_cls=False):
        
        assert len(x.shape)==4
        b, _, _, _ = x.shape

        x = self.m(x)

        out = self.conv_out(x).view(b, 1, 1, 1)
        if out_cls:
            out_cls = self.conv_out_cls(x).view(b, -1)
            return out, out_cls
        else:
            return out

class ud2d(nn.Module):
    def __init__(self, args, input_channel=2):
        super(ud2d, self).__init__()

        self.encoder = Encoder(args, input_channel=input_channel)
        self.decoder = Decoder(args)
        self.args = args

        self.mse_loss = nn.MSELoss()

        self.apply(weights_initialize_xavier_normal)

    def forward(self, x):
        concat_feat, encoded = self.encoder(x, layer_out=True)

        y = self.decoder(encoded, concat_feat=concat_feat)

        return y

class Encoder(nn.Module):
    def __init__(self, args, input_channel=1):
        super(Encoder, self).__init__()
        self.use_norm = args.encoder_norm
        self.input_channel = input_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode = args.batch_mode
        k=3
        p=1

        c1 = args.initial_channel
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use, kernel=3, padding=1)
        self.l11 = CBR(in_channel=c1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use,kernel=3, padding=1)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)

        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)

        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)

        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        self.l51 = CBR(in_channel=c5, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)

        self.mpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, layer_out=False):

        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        latent = self.l51(self.l50(l4_pool))

        if layer_out:
            return [l4, l3, l2, l1], latent

        else:
            return latent

class Decoder(nn.Module):
    def __init__(self, args, output_channel=1, skip=False):
        super(Decoder, self).__init__()
        self.use_norm = args.decoder_norm
        self.output_channel = output_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode = args.batch_mode
        self.args = args

        k=3
        p=1

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2

        self.conv_module = nn.ModuleList()
        self.upsampling_module = nn.ModuleList()
        if args.concat_feat:
            self.conv_in = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                               batch_mode=self.batch_mode)
            for idx, (i, j) in enumerate(zip([c5, c4, c3, c2], [c4, c3, c2, c1])):
                self.conv_module.append(nn.Sequential(CBR(in_channel=i, out_channel=j, use_norm=self.use_norm,
                                                          lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                                      CBR(in_channel=j, out_channel=j, use_norm=self.use_norm,
                                                          lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
                                                      ))
                if idx == 0:
                    self.upsampling_module.append(
                        nn.ConvTranspose2d(in_channels=j, out_channels=j, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))

                else:
                    self.upsampling_module.append(nn.ConvTranspose2d(in_channels=i, out_channels=j, kernel_size=(2,2), stride=(2,2), padding=(0,0)))



        self.out = nn.Conv2d(in_channels=args.initial_channel, out_channels=3, kernel_size=(1, 1),
                             stride=(1, 1), padding=(0, 0))

        self.activation = nn.LeakyReLU() if self.lrelu_use else nn.ReLU()


    def forward(self, latent, concat_feat=None):

        x = self.conv_in(latent)

        if self.args.concat_feat:
            for idx, (conv, up) in enumerate(zip(self.conv_module, self.upsampling_module)):
                x = up(x)
                x = torch.cat([x, concat_feat[idx]], dim=1)
                x = conv(x)

        y = self.out(x)
        return y

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, use_norm=True, kernel=3, stride=1
                 , lrelu_use=False, slope=0.1, batch_mode='I', sampling='down', rate=1):
        super(CBR, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_norm = use_norm
        self.lrelu = lrelu_use

        if sampling == 'down':
            self.Conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(kernel, kernel), stride=(stride, stride),
                                  padding=padding, dilation=(rate, rate))
        else:
            self.Conv = nn.ConvTranspose2d(self.in_channel, self.out_channel, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        if self.use_norm:
            if batch_mode == 'I':
                self.Batch = nn.InstanceNorm2d(self.out_channel)
            elif batch_mode == 'G':
                self.Batch = nn.GroupNorm(self.out_channel//16, self.out_channel)
            else:
                self.Batch = nn.BatchNorm2d(self.out_channel)

        if self.lrelu:
            self.activation = nn.LeakyReLU(negative_slope=slope)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):

        if self.use_norm:
            out = self.activation(self.Batch(self.Conv(x)))
        else:
            out = self.activation(self.Conv(x))

        return out


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

def weights_initialize_xavier_normal(m):

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)