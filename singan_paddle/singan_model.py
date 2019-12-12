#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import division
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC, SpectralNorm
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import ParamAttr
import numpy as np
import math
import os

#param_attr=ParamAttr(name="%s_conv_w"%name_scope, initializer = fluid.initializer.Xavier()),

class SpectralConv(fluid.dygraph.Layer):
    def __init__(self, name_scope, in_channel, out_channel, opt):
        super(SpectralConv, self).__init__(name_scope)
        self._conv = Conv2D("%s_conv"%(name_scope),
             num_filters=out_channel,
             filter_size=opt.ker_size,
             padding = opt.padd_size,
             stride = 1)
        self.spectralNorm = SpectralNorm('%s_sn'%(name_scope), dim=1, power_iters=1)
        #self.weight = self._conv.create_parameter(shape=[out_channel,in_channel, opt.ker_size, opt.ker_size], dtype='float32', attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=0.02, seed=0)))
        #self.weight.stop_gradient=False

    def forward(self, x):
        #self.weight = self.spectralNorm(self.weight)
        y = self._conv(x)
        return y 
         

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self, name_scope, in_channel, out_channel, opt):
        super(ConvBNLayer, self).__init__(name_scope)
        self._conv = SpectralConv(name_scope, in_channel, out_channel, opt)
        self._bn = BatchNorm("%s_bn"%name_scope,
             num_channels = out_channel,
             param_attr=ParamAttr(name="%s_conv_w"%name_scope, initializer=fluid.initializer.Normal(loc=1.0, scale=0.02, seed=0)),
             bias_attr=ParamAttr(name="%s_bn_b"%name_scope, initializer=fluid.initializer.ConstantInitializer(value=0.0)),
             )

    def forward(self, x):
        y = self._conv(x)
        y = self._bn(y)
        y = fluid.layers.leaky_relu(y)
        return y

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope, opt):
        super(Discriminator, self).__init__(name_scope)
        self.layers = []
        in_channel = 3
        self.pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        for i in range(0, opt.num_layer-1):
            out_channel = max(opt.min_nfc, int(opt.nfc/math.pow(2,i)))
            convbn = ConvBNLayer("%s_%d"%(name_scope, i), in_channel, out_channel, opt)
            self.add_sublayer("%s_%d"%(name_scope, i), convbn)
            self.layers.append(convbn)
            in_channel = out_channel
        conv = SpectralConv(name_scope, in_channel, 1, opt)
        self.add_sublayer("%s_%d_conv"%(name_scope, i+1), conv)
        self.layers.append(conv)
    def forward(self, x):
        y = x
        #y = fluid.layers.pad(x, [0,0,0,0,self.pad_noise,self.pad_noise,self.pad_noise,self.pad_noise])
        for layer in self.layers:
            y = layer(y)
        y = fluid.layers.sigmoid(y)
        return y

class Generator(fluid.dygraph.Layer):
    def __init__(self, name_scope, opt):
        super(Generator, self).__init__(name_scope)
        self.layers = []
        in_channel = 3
        self.pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        
        for i in range(0, opt.num_layer-1):
            out_channel = max(opt.min_nfc, int(opt.nfc/math.pow(2,i)))
            convbn = ConvBNLayer("%s_%d"%(name_scope, i), in_channel, out_channel, opt)
            self.add_sublayer("%s_%d"%(name_scope, i), convbn)
            self.layers.append(convbn)
            in_channel = out_channel
        conv = SpectralConv(name_scope, in_channel, opt.nc_im, opt)
        self.add_sublayer("%s_%d"%(name_scope, i+1), conv)
        self.layers.append(conv)
    def forward(self, x, y):
        #print("x",x.shape, "y",y.shape)
        #x = fluid.layers.pad(x, [0,0,0,0,self.pad_noise,self.pad_noise,self.pad_noise,self.pad_noise])
        #print("x",x.shape, "y",y.shape, "pad",self.pad_noise)
        for layer in self.layers:
            x = layer(x)
        x = fluid.layers.tanh(x)
        #print("x",x.shape, "y",y.shape)
        x = fluid.layers.image_resize(x, out_shape=y.shape[2:4], resample='BILINEAR')
        z = fluid.layers.elementwise_add(x, y)
        return z
         
"""
class SinGan(object):
    def __init__(self, opt, is_training=True):
        self.opt = opt
        self.is_training = is_training
    def ConvBlock(self, data, prefix, out_channel, ker_size, padd, stride):
        conv = Conv2D(input=data, num_filters=out_channel, filter_size=ker_size, padding=padd, stride=stride,
           param_attr=ParamAttr(name=prefix + "_weights", initializer=fluid.initializer.Normal(loc=0.0, scale=0.02)), 
           name=prefix + "_conv")
        bn = BatchNorm(input=conv, act='leaky_relu', is_test=(not self.is_training), 
           param_attr=ParamAttr(name=prefix + "_weights", initializer=fluid.initializer.Normal(loc=1.0, scale=0.02)), 
           bias_attr=ParamAttr(name=prefix + "_bias", initializer=fluid.initializer.ConstantInitializer(value=0.0)), 
           name=prefix + "_bn")
        return bn
    def network(self, image, name='GA'):
        net = to_variable(image)
        opt = self.opt
        for i in range(0, opt.num_layer-1):
            out_channel = int(opt.nfc/math.pow(2,i)) 
            net = self.ConvBlock(data=net, prefix="%s_%d"%(name, i), out_channel=max(out_channel, opt.min_nfc), ker_size=opt.ker_size, padd=opt.padd_size, stride=1)
        if name == 'GA':
            net = Conv2D(input=net, prefix="%s_conv"%name, num_filters=opt.nc_im, kernel_size=opt.ker_size,stride =1, padding=opt.padd_size, act='tanh')
        else:
            net = Conv2D(input=net, prefix="%s_conv"%name, num_filters=opt.nc_im, kernel_size=opt.ker_size,stride =1, padding=opt.padd_size)
        return net

"""
            
