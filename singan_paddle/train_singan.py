#!/usr/bin/env python
#-*- coding=utf8 -*-

from config import get_arguments
import traceback
import cv2
import os
import numpy as np
from singan_model import Generator, Discriminator
#from models.model import Generator, Discriminator
from utils import creat_reals_pyramid, post_config, generate_noise, resize, dump_img
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

def train(opt):
    images = creat_reals_pyramid(opt)
    place = fluid.CUDAPlace(0) if opt.use_gpu else fluid.CPUPlace()
    priors = []
    prior_recons = []
    netD_arrs = []
    netG_arrs = []
    noiseamp_arrs = []
    opt.padd_size = 1
    for idx in range(0,len(images)):
        outdir = "%s/%d/"%(opt.out, idx)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        with fluid.dygraph.guard():
            real = images[idx]
            in_s = np.zeros(shape=real.shape, dtype=np.float32)
            zero = fluid.layers.zeros(shape=[1], dtype='float32')
            #zero.stop_gradient = True
            one = fluid.layers.ones(shape=[1], dtype='float32')
            #one.stop_gradient = True
            alpha = to_variable(np.array([opt.alpha]).astype('float32')) 
            optimizerG = fluid.optimizer.Adam(learning_rate=opt.lr_d, beta1=opt.beta1, beta2=0.999, name='net_GA')
            optimizerD = fluid.optimizer.Adam(learning_rate=opt.lr_d, beta1=opt.beta1, beta2=0.999, name='net_DA')
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            #optimizerD = fluid.optimizer.RMSPropOptimizer(learning_rate=opt.lr_d, name="opD")
            #optimizerG = fluid.optimizer.RMSPropOptimizer(learning_rate=opt.lr_g, name="opG")
            #fluid.clip.set_gradient_clip(fluid.clip.GradientClipByValue(0,1))
            netD = Discriminator("DA", opt)
            netG = Generator("GA", opt)
           # fluid.clip.set_gradient_clip(fluid.clip.GradientClipByValue(min=-0.01, max=0.01),param_list=[netD.parameters(),netG.parameters()])
            vreal = to_variable(real)
            for epoch in range(opt.niter): 
                noise_epoch = generate_noise(real.shape, opt)
                prev = in_s 
                prev_rec = in_s
                opt.noise_amp = 1
                for idx in range(len(netG_arrs)):
                    prev = priors[idx]
                    prev_rec = prior_recons[idx]
                    opt.noise_amp = noiseamp_arrs[idx]
                prev = resize(prev, (real.shape[3], real.shape[2]))
                prev_rec = resize(prev_rec, (real.shape[3], real.shape[2]))
                vprev = to_variable(prev)
                vprev_rec = to_variable(prev_rec)
                for j in range(opt.Dsteps):
                    netD.clear_gradients()
                    outD_real = netD(vreal)
                    errD_real = fluid.layers.mean(outD_real)
                    errD_real = 0.0 - errD_real
                    errD_real.backward(backward_strategy)
                    #errD_real = fluid.layers.elementwise_sub(zero, errD_real)
#                    errD_real.backward(backward_strategy)
                    noise = opt.noise_amp * noise_epoch + prev
                    vnoise = to_variable(noise)
                    outG_fake = netG(vnoise.detach(), vprev)
                    outD_fake = netD(outG_fake.detach())
                    errD_fake = fluid.layers.mean(outD_fake)
            #        errD_fake = fluid.layers.elementwise_sub(zero, errD_fake)
                    errD_fake.backward(backward_strategy)
                    #gradient_penalty = calc_gradient_penalty(netD, vreal, outG_fake, opt, backward_strategy)
                    #gradient_penalty.backward()
                    #errD = errD_real + errD_fake + gradient_penalty
                    errD = errD_real + errD_fake
                    params_d = optimizerD.backward(errD, parameter_list=netD.parameters())
                    optimizerD.apply_gradients(params_d)
                for j in range(opt.Gsteps):
                    netG.clear_gradients()
                    outD_fakeG = netD(outG_fake)
                    errD_fakeG = 0.0 - fluid.layers.mean(outD_fakeG)
       #             errD_fakeG = fluid.layers.elementwise_add(zero, errD_fakeG)
                    errD_fakeG.backward(backward_strategy)
                    noise_fake = opt.noise_amp *noise_epoch + prev_rec 
                    noise_fake = to_variable(noise_fake)
                    outG_fake_rec = netG(noise_fake.detach(), vprev_rec)
                    rec_loss = fluid.layers.mse_loss(vreal, outG_fake_rec)
                    RMSE = fluid.layers.sqrt(rec_loss).numpy()
                    rec_loss = fluid.layers.elementwise_mul(alpha, rec_loss)
                    rec_loss.backward(backward_strategy) 
                    errG =  rec_loss + errD_fakeG
                    #errG =  errD_fakeG
                    params_g = optimizerG.backward(errG, parameter_list=netG.parameters())
                    optimizerG.apply_gradients(params_g)
                #netD.clear_gradients()
                #netG.clear_gradients()
                if epoch % 25 == 0 or epoch == (opt.niter-1):
                    print('shape %s [epoch:%d/%d][errD:%.5f][errG:%.5f][rec_loss:%.5f][noise_amp:%.5f][errD_real:%.5f][errD_fake:%.5f][outD_fakeG:%.5f]' %
                         (real.shape, epoch, opt.niter, errD.numpy(), errG.numpy(), rec_loss.numpy(),
                          opt.noise_amp, errD_real.numpy(), errD_fake.numpy(), errD_fakeG.numpy()))
                if epoch % 500 == 0 or epoch == (opt.niter-1):
                    dump_img(outG_fake.numpy(), os.path.join(outdir, "fake_sample_%d_%s"%(epoch, opt.input_name)))
                    dump_img(outG_fake_rec.numpy(), os.path.join(outdir, "G(z_opt)_%d_%s"%(epoch, opt.input_name)))
            fluid.dygraph.save_dygraph(netD.state_dict(), os.path.join(outdir, "DA"))
            fluid.dygraph.save_dygraph(netG.state_dict(), os.path.join(outdir, "GA"))
            fluid.dygraph.save_dygraph(optimizerD.state_dict(), os.path.join(outdir, "DA"))
            fluid.dygraph.save_dygraph(optimizerG.state_dict(), os.path.join(outdir, "GA"))
            opt.noise_amp = opt.noise_amp_init*RMSE
            netD.eval()
            netD_arrs.append(netD)
            netG.eval()
            netG_arrs.append(netG)
            priors.append(outG_fake.numpy())
            prior_recons.append(outG_fake_rec.numpy())
            noiseamp_arrs.append(opt.noise_amp)

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name', default='cows.png')
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    post_config(opt)
    try:
        train(opt)
    except Exception as e:
        traceback.print_exc(e)
