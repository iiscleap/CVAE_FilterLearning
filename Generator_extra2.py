import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['VAE']


class Encoder(torch.nn.Module):
    def __init__(self, param):
        super(Encoder, self).__init__()
        self.input_nc = param['input_nc']
        self.output_nc = param['output_nc']
        self.ngf = param['ngf']
        self.gpu_ids = param['gpu_ids']
        self.filt_h = param['filt_h']
        self.filt_w = param['filt_w']
        self.batch_size = param['batch_size']

	self.r1 = torch.nn.Parameter(torch.randn(self.filt_h).cuda())
        self.s1 = torch.nn.Parameter(torch.randn(self.filt_w).cuda())
        self.filt1 = torch.ger(self.r1, self.s1)
        self.f1 = (self.filt1.resize(self.output_nc, self.input_nc, self.filt1.shape[0], self.filt1.shape[1]))

        self.r2 = torch.nn.Parameter(torch.randn(self.filt_h).cuda())
        self.s2 = torch.nn.Parameter(torch.randn(self.filt_w).cuda())
        self.filt2 = torch.ger(self.r2, self.s2)
        self.f2 = (self.filt2.resize(self.output_nc, self.input_nc, self.filt2.shape[0], self.filt2.shape[1]))

	self.f = torch.cat((self.f1, self.f2), 0)  # weight tensor has to be (self.ngf X self.input_nc X H X W)	

        self.fc = None

        self.fnn_layer1 = torch.nn.Linear(param['num_nodes_fnn'][0], param['num_nodes_fnn'][1])
        self.fnn_layer2 = torch.nn.Linear(param['num_nodes_fnn'][1], param['num_nodes_fnn'][2])
        self.tanh = torch.nn.Tanh()
	self.relu = torch.nn.ReLU()
	self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = (self.tanh(F.conv2d(x, self.f, padding=(2, 2))))
	out = out1.view(self.ngf * self.batch_size, -1)
        out = (self.tanh(self.fnn_layer1(out)))
        out = (self.tanh(self.fnn_layer2(out)))
        return out


class Decoder(torch.nn.Module):
    def __init__(self, param):
        super(Decoder, self).__init__()
	self.ngf = param['ngf']
        self.batch_size = param['batch_size']
	self.patch_length = param['patch_length']
	self.bands = param['bands']
	self.output_nc = param['output_nc']
	self.filt_h = param['filt_h']
        self.filt_w = param['filt_w']
        self.fnn_layer3 = torch.nn.Linear(param['num_nodes_mean_var'], param['num_nodes_fnn'][2])
        self.fnn_layer4 = torch.nn.Linear(param['num_nodes_fnn'][2], param['num_nodes_fnn'][3])
        self.fnn_layer5 = torch.nn.Linear(param['num_nodes_fnn'][3], param['num_nodes_fnn'][4])
	self.output_layer = torch.nn.ConvTranspose2d(self.ngf, self.output_nc,
                                                     kernel_size=(self.filt_h, self.filt_w), padding=(2,2), bias=False)
        self.tanh = torch.nn.Tanh()
	self.relu = torch.nn.ReLU()
	self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = (self.tanh(self.fnn_layer3(x)))
        out = (self.tanh(self.fnn_layer4(out)))
        out = (self.tanh(self.fnn_layer5(out)))
        out = out.view(self.batch_size, self.ngf, self.patch_length, self.bands)
	out = (self.output_layer(out))
        return out


class VAE(torch.nn.Module):

    def __init__(self, param):
        super(VAE, self).__init__()
        self.encoder = Encoder(param)  # rank-1 2D kernels in both conv layers
        self.decoder = Decoder(param)
        self._enc_mu = torch.nn.Linear(param['num_nodes_fnn'][2], param['num_nodes_mean_var'])
        self._enc_log_sigma = torch.nn.Linear(param['num_nodes_fnn'][2], param['num_nodes_mean_var'])

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, inp):
        h_enc = self.encoder(inp)
        z = self._sample_latent(h_enc)
        out = self.decoder(z)
        return h_enc, out.view(inp.shape)

