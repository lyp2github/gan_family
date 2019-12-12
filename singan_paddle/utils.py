#!/usr/bin/env python
#-*- coding=utf8 -*-
import math
import os
import sys
import cv2
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
from sklearn import preprocessing

def img_read(imgfile):
    image = cv2.imread(imgfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def calc_scale(image, opt):
    shapes = []
    (height, width, channel) = image.shape
    minsize = min(width, height)
    maxsize = max(width, height)
    opt.num_scales = int((math.log(math.pow(opt.min_size / minsize, 1), opt.scale_factor))) + 1
    scale2stop = int(math.log(min([opt.max_size, maxsize]) / maxsize, opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop 
    opt.scale1 = min(opt.max_size / maxsize,1)
    opt.scale_factor = math.pow(opt.min_size / (minsize), 1 / (opt.stop_scale))
    for i in range(0, opt.stop_scale + 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale-i)
        w = int(width * scale)
        h = int(height * scale)
        shapes.append((w, h))
    return shapes

def resize(data, sp, inter=cv2.INTER_CUBIC):
    data = data[0].transpose(1, 2, 0)
    #data = preprocessing.MinMaxScaler().fit_transform(data)
    data = data * 255 #[-1,1] -> [0,255]
    data = data.astype(np.uint8)
    #sp = [width, height]
    data = cv2.resize(data, sp, interpolation = inter)
    data = preprocess_image(data)
    return data

def preprocess_image(timg):
    timg = timg[:, :, :, None]
    timg = timg.transpose((3, 2, 0, 1))/255.0 # [0,255] -> [-1, 1]
    return timg.astype('float32')

def creat_reals_pyramid(opt):
    imgfile = os.path.join(opt.input_dir, opt.input_name)
    img = img_read(imgfile)
    images = []
    shapes = calc_scale(img, opt)
    for sp in shapes:
        timg = cv2.resize(img, sp, interpolation = cv2.INTER_CUBIC)
        images.append(preprocess_image(timg))
    return images

def post_config(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
        np.random.seed(opt.manualSeed)
    opt.noise_amp_init = opt.noise_amp
    opt.out = os.path.join(opt.out, opt.input_name.split(".")[0])
    if os.path.isdir(opt.out):
        os.mkdir(opt.out)
def generate_noise(shape, opt):
    out = np.random.normal(loc=0.0, scale=1, size=shape)
#    out = np.random.randn(shape[2],shape[3])
#    out = np.tile(out, (1,3,1,1))
    #out = np.random.randn(*shape)
    return out.astype('float32')

def dump_img(image, imgpath):
    dumpimg = image[0]*255
    dumpimg = dumpimg.transpose((1,2,0))
    #image = image[-1,:,:,:]
    #image = image.transpose((1,2,0)) 
    #plt.imsave(imgpath, image, vmin=0, vmax=1)
    dumpimg = cv2.cvtColor(dumpimg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgpath, dumpimg) 

def dump_gif(imagearr, gif_name):
    frames = []
    TIME_GAP = 0.075
    idx = 0
    for image in imagearr:
        dumpimg = image[0]*255
        dumpimg = dumpimg.transpose((1,2,0))
        #dumpimg = cv2.cvtColor(dumpimg, cv2.COLOR_RGB2BGR)
        frames.append(dumpimg)
        dumpimg = cv2.cvtColor(dumpimg, cv2.COLOR_RGB2BGR)
        cv2.imwrite("%s_%d.png"%(gif_name, idx), dumpimg)
        idx += 1
    imageio.mimsave(gif_name, frames, 'GIF', duration = TIME_GAP)  
