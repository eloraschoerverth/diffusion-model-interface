import math
from inspect import isfunction
from functools import partial
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import sklearn.metrics 

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.optim import Adam

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import cv2

def get_layer_names(model):
    layer_names = []
    for name, layer in model.named_modules():
        for layer_definition in [nn.Conv2d]:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)
                    
    return layer_names   

def normalize(val):
    return (val-val.min())/(val.max()-val.min())

def max_sum_channel(data, img_mask = None):
    
    if img_mask is not None:
        #scale mask to size of activation layer
        c, h, w = data.shape

        img_mask = np.asarray(img_mask["mask"])[:,:,0:1]
        img_mask = cv2.resize(img_mask, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        img_mask[img_mask > 0] = 1
        #muliply mask with activations to mask area and compute localized max layer
        data = data*img_mask

    layer_s = data.sum(axis=(1,2))

    return np.argmax(layer_s).item()
   
def interpolate_attention_map(activations, sample_steps, size=64, sample_id = 0):  
        amaps = []
        for i in sample_steps:
            resized_activations = []
            for feats in [torch.Tensor(val) for key, val in activations[str(i)].items()]:
                feats = feats[sample_id][None]
                feats = nn.functional.interpolate(
                    feats, size=size, mode="bilinear"
                )
                resized_activations.append(feats[0])
            am = np.concatenate(resized_activations, axis=0).sum(axis=0)
            amaps.append(normalize(am))
        return amaps
    
def set_analysis_settings(sample_steps, layers, track_mode, mask, s_y, s_t):
    
    return  {
            "sample_steps": sample_steps,
            "layers": layers,
            "track_mode": track_mode,
            "mask": mask,
            "s_y": s_y,
            "s_t": s_t}
                    
    

    
    
    
    
    
    