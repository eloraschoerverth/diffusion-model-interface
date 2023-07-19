import math
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from torchmetrics.functional import structural_similarity_index_measure

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.optim import Adam

#Beta schedules
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        mode = 'unc',
        image_size = 32,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        device='cuda'
    ):

        super().__init__()
        self.mode = mode
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.device = device
        self.betas = betas
        self.timesteps = timesteps

        device = next(model.parameters()).device

        # define alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)    

    def extract(self,a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    #Marco
    def label_reshaping(self, y, b, h, w, device):
            y = y[:,None] if y.ndim == 1 else y
            assert y.ndim == 2, f'conditions shape {y.shape} should be (batch, n_conditions)'
            assert torch.is_tensor(y), 'labels array should be a pytorch tensor'
            n_conditions = y.shape[-1]
            labels = torch.ones((b,n_conditions,h,w)).to(device)
            return torch.einsum('ijkl,ij->ijkl', labels, y)

    def label_concatenate(self,x,y):
            return torch.cat([x,y],dim=1)

    def p_losses(self, x_start, t, noise=None, loss_type="l1", y = None):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.mode == 'c':
            x_noisy = self.label_concatenate(x_noisy, y)

        predicted_noise = self.model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
            
        mse = F.mse_loss(noise, predicted_noise)
        psnr = 10 * math.log10(255 / mse.item())
        
        ssim = structural_similarity_index_measure(predicted_noise, noise)
            

        return loss, psnr, ssim
    
    
    def p_g_sample(self, x, t, t_index, mask = None):
        
        noise = torch.randn_like(x[:,:self.channels])
        input_grads = None
        
        betas_t = self.extract(self.betas, t, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x[:,:self.channels].shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x[:,:self.channels].shape)
       
        input = x.detach()
        input.requires_grad = True

        prediction =  sqrt_recip_alphas_t * (
            input[:,:self.channels] - betas_t * self.model(input, t) / sqrt_one_minus_alphas_cumprod_t
        )

        model_mean = prediction

        if t_index == 0 :
            return model_mean, input_grads 

        posterior_variance_t = self.extract(self.posterior_variance, t, x[:,:self.channels].shape)

        #initialize grad tensor but flatten it
        b,c,h,w = x.shape
        #Allocate for only one channel to deal with memory error in RGB images
        input_grads = numpy.empty(shape = (h*w,1,h,w))


        pred = prediction if mask is None else prediction * mask
        
        #If prediction has more than one color channels, sum them up
        #The gradient is then calculated for all color channels of each pixel at once
        if self.channels > 1:
            pred = torch.sum(pred[0],0)
            
              
        for k, f_val in enumerate(pred.flatten()):
            #if value has been masked, continue with next one
            if f_val is None or f_val == 0:   
                continue

            if input.grad is not None:
                input.grad.data.zero_()

            f_val.backward(retain_graph=True)
            
            if input.grad is not None:
                g = input.grad.data.cpu().numpy()
                g = g.reshape(c,h,w)
                input_grads[k] = numpy.sum(g, axis=0)
                input.grad.data.zero_()
                
        return (model_mean + torch.sqrt(posterior_variance_t) * noise), input_grads


    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x[:,:self.channels].shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x[:,:self.channels].shape)

        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
            x[:,:self.channels] - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0 :
            return model_mean
            
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x[:,:self.channels].shape)
            noise = torch.randn_like(x[:,:self.channels])

            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            #extract output and standardize
            self.l_features[layer_id] = output.detach().cpu().numpy()
            
        return fn
    
    def save_gradient_hook(self, layer_id: str):
        def fn(module, grad_in, grad_out):
            #extract output and standardize
            self.l_gradients[layer_id].append(grad_out[0].detach().cpu().numpy())

        return fn

    # Algorithm 2 (including returning all images)
    def p_sample_loop(self, shape, random_seed = None, y=None, analysis = None):

        b, c, h, w = shape
         
        # sample random starting noise.
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if b == 1:
                img = torch.randn(shape, device=self.device)
            elif b > 1:
                noise = torch.randn((1,c,h,w), device=self.device)
                img = noise.repeat(b,1,1,1)
        else:
            img = torch.randn(shape, device=self.device)
            
        imgs, handles, gv = [], [], []
        
        activations, gradients, grad_volume  = {}, {}, {}
            
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            self.l_features = {layer: torch.empty(0) for layer in analysis["layers"]}
            self.l_gradients = {layer: [] for layer in analysis["layers"]}
            
            if analysis["s_t"] is not None and analysis["s_t"] == i:
                #swap label
                y = analysis["s_y"]
            
            #conditioning
            if self.mode == 'c':
                if len(y.shape) == 1: # Labels 1D
                    y = self.label_reshaping(y, b, self.image_size, self.image_size, self.device)
                img = self.label_concatenate(img,y)
             
            if i in analysis["sample_steps"]:
                #attach activation hook
                if analysis["track_mode"] == 0:
                    print("Tracking activations at timestep ",i)
                    for layer_id in analysis["layers"]:
                        layer = dict([*self.model.named_modules()])[layer_id]
                        h = layer.register_forward_hook(self.save_outputs_hook(layer_id))
                        handles.append(h)
                        
                    with torch.no_grad():
                        img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), i)

                #gradient sampling  
                elif analysis["track_mode"] == 1:
                    print("Tracking gradient at timestep ",i)
                    for layer_id in analysis["layers"]:
                        layer = dict([*self.model.named_modules()])[layer_id]
                        h = layer.register_backward_hook(self.save_gradient_hook(layer_id))
                        handles.append(h)
                    img, gv = self.p_g_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), i, analysis["mask"])

            #default sampling
            else:
                with torch.no_grad():
                    img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), i)

            #write images, gradients and activations to lists
            imgs.append(img.detach().cpu().numpy())
            
            activations[str(i)] =  self.l_features                
            gradients[str(i)] = self.l_gradients
            grad_volume[str(i)] = gv

            
            #Remove activation handles after sampling
            if handles != []:
                for h in handles:
                    h.remove()
                handles = []
            
        return imgs, grad_volume, activations, gradients 

    
    def sample(self, batch_size=1, random_seed = None, condition=None, analysis = None):
        #assert self.model.channels <= 2, f'Methods have only been implemented for greyscale models'
        #for now limit to one sample if gradient is sampled. Issue with slow runtime and formatting masked samples  
        
        if analysis["track_mode"] == 1:
            assert batch_size == 1, f'Batch size must not exceed 1 if sampling gradient'
                    
        return self.p_sample_loop(shape=(batch_size, self.channels, self.image_size, self.image_size), random_seed = random_seed, y=condition, analysis = analysis)
