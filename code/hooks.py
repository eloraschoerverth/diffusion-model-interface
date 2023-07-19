import regex as re
import math
from inspect import isfunction
from functools import partial
import numpy as np
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.utils.data as data


class ObscureChannelHook():
    def __init__(self, module, channel = 0):
        self.channel = channel
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        #set one channel to zero
        #output[:,self.channel,:] = torch.sign(torch.zeros_like(output[:,self.channel,:]))    
        output[:,self.channel,:] = torch.zeros_like(output[:,self.channel,:]) 
        return output
    
    def close(self):
        print("Removing hook")
        self.hook.remove()
        
class ObscurePixelHook():
    def __init__(self, module, x, y):
        self.hook = module.register_forward_hook(self.hook_fn)  
        self.x = x
        self.y = y
        

    def hook_fn(self, module, input, output):
        b,c, h,w = output.shape
        assert x <= w, f'X Pixel index is too above layer dimension'
        assert y <= h, f'Y Pixel index is too above layer dimension'
        #set one pixel to zero
        output[:,:,x,y] = 0
        
        return output
    
    def close(self):
        print("Removing hook")
        self.hook.remove()

class ObscureChannelMaskHook():
    def __init__(self, module, mask,channel):
        self.mask = mask
        self.channel = channel
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        #conditioning channel
        output[0,self.channel,:,:] = output[0,self.channel,:,:]*self.mask
        return output
    
    def close(self):
        print("Removing hook")
        self.hook.remove()
        
        
class ObscureMaskHook():
    def __init__(self, module, mask):
        self.mask = mask
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
#         output[:,self.channel,:] = torch.zeros_like(output[:,self.channel,:])
        return output * self.mask
    
    def close(self):
        print("Removing hook")
        self.hook.remove()
        

def layer_keys(key):
    key = re.sub(r"\d{1}", r"[\g<0>]", key)
    key = key.replace(".[","[") 
    return key