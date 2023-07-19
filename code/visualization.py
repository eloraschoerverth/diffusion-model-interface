import math
from inspect import isfunction
from functools import partial
import numpy as np
import sklearn.metrics 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

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
import plotly.express as px
from plotly.subplots import make_subplots

def normalize(value):
    
    return (value-value.min())/(value.max()-value.min())


def vs_dist_fig(values1, values2, mode, layers, sample_steps, label1, label2):
    """
        Plots distance between layers of two labels over time

            Parameters:
            ----------
            values1 : ndarray
                3D array containing either activations or gradients of first label.
            values2 : ndarray
                3D array containing either activations or gradients of second label
            mode: integer
                Flag indicating whether activation or gradient distances are visualized.
            sample_steps: list
                List contains timesteps where values where drawn from the model
            label1: string
                Label of the first sample.
            label2: string
                Label of the second sample

        Returns:
            fig: go.Figure
                Plotly Graph Objects Figure for the distances.
        --------     
    """
    
    x_values = sample_steps
    layer_distances = []
    for i, l in enumerate(layers):
        dist = []

        for step in sample_steps:
                
            if mode == 0:
                v1 = normalize(values1[str(step)][l][0])
                v2 = normalize(values2[str(step)][l][0])
            elif mode == 1:
                v1 = normalize(np.stack(values1[str(step)][l], axis=0).sum(axis=0)[0])
                v2 = normalize(np.stack(values2[str(step)][l], axis=0).sum(axis=0)[0])
            dist.append(np.linalg.norm(v1 - v2))

        layer_distances.append(dist)

    x = sample_steps
    fig = go.Figure(
        data = [
            go.Scatter(x=x_values, y=layer_distances[0], name=layers[0]),
            go.Scatter(x=x_values, y=layer_distances[1], name=layers[1]),
            go.Scatter(x=x_values, y=layer_distances[2], name=layers[2]),
            go.Scatter(x=x_values, y=layer_distances[3], name=layers[3]),
            go.Scatter(x=x_values, y=layer_distances[4], name=layers[4]),
            go.Scatter(x=x_values, y=layer_distances[5], name=layers[5]),
        ],
        layout = {"xaxis": {"title": "Sample Step"}, "yaxis": {"title": "Euclidean Distance"}})
    
  
        
    if mode == 0:
        fig.update_layout(
            title_text = "Euclidean distance between activations of samples "+str(label1.item())+" and " +str(label2.item()), 
            template="simple_white", 
            legend=dict(
                title=None, orientation="h", y=2, yanchor="bottom", x=0.5, xanchor="center"
            )
        )
    elif mode == 1:
        fig.update_layout(
            title_text = "Euclidean distance between gradients of two samples "+str(label1.item())+" and " +str(label2.item()),
            template="simple_white", 
            legend=dict(
                title=None, orientation="h", y=2, yanchor="bottom", x=0.5, xanchor="center"
            )
        )
    
    fig.update_layout(
        height=500,
        width=700,
    )
    #create border around plot
    fig.update_xaxes(autorange='reversed',showline=True, linewidth=1, showgrid=True, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=True, mirror=True)
    
    return fig


def gradient_volume_fig(grad_vol, sample_steps, im_shape):
    """
        Plots the gradient volume at a specified timestep.
        
        Parameters:
        ----------
        grad_vol : list(ndarray)
            list containing gradient volume for selected timesteps
        sample_steps : list
            Contains timesteps where the gradient volume was drawn.
        im_shape: integer
            Max value of height/width of the output image.

        Returns:
        fig: go.Figure
            Plotly Graph Objects Figure depicting the gradient volume for each sample step.
        --------     
    """
    
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in sample_steps:
        gv = grad_vol[str(step)][:,0,:,:]
        z1,x1,y1 = np.where(gv>gv.mean()+1e-3)
        
       

        fig.add_trace(
            go.Scatter3d(
                visible=False,
                name= "Vol " + str(step),
                x=z1, 
                y=y1, 
                z=x1,
                mode='markers'
            ))


    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            label=str(sample_steps[i]),
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Gradient volume at step: " + str(sample_steps[i])}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        steps=steps,
        pad=dict(b=30,l=60,r=60,t=10),
    )]
    
    fig.update_traces(marker_size = 4)


    fig.update_layout(
        sliders=sliders,

        scene=dict(
            xaxis_title='Pixel-wise',
            yaxis_title='Image X Axis',
            zaxis_title='Image Y Axis',
            yaxis = dict(range=[0,im_shape],),
            zaxis = dict(range=[im_shape,0]),

        )
    )

    return fig

def dist_fig(values, mode, layers, sample_steps):
    """
        Plots distance figure of either activations or gradients with respect to their previous timestep.
        
        Parameters:
        ----------
        values: ndarray
        mode: integer
        layers: list(str)
        sample_steps: list(int)

        Returns:
        fig: go.Figure
            Plotly Graph Objects Figure.
        --------     
    """
    
    x_values = sample_steps
    layer_dists = []    
    
    for l in layers:
        dist = []
        for k, step in enumerate(sample_steps):
            
            if k == 0:
                continue
            if mode == 0:           
                s1 = normalize(values[str(sample_steps[k-1])][l][0])
                s2 =normalize(values[str(sample_steps[k])][l][0])
            elif mode == 1:
                s1 = normalize(np.stack(values[str(sample_steps[k-1])][l], axis=0).sum(axis=0)[0])
                s2 = normalize(np.stack(values[str(sample_steps[k])][l], axis=0).sum(axis=0)[0])
            dist.append(np.linalg.norm(s1 - s2))
                
        layer_dists.append(dist)

    fig = go.Figure(
        data = [
            go.Scatter(x=x_values[1:], y=layer_dists[0], name=layers[0]),
            go.Scatter(x=x_values[1:], y=layer_dists[1], name=layers[1]),
            go.Scatter(x=x_values[1:], y=layer_dists[2], name=layers[2]),
            go.Scatter(x=x_values[1:], y=layer_dists[3], name=layers[3]),
            go.Scatter(x=x_values[1:], y=layer_dists[4], name=layers[4]),
            go.Scatter(x=x_values[1:], y=layer_dists[5], name=layers[5]),
        ],
        layout = {"xaxis": {"title": "Sample Step"}, "yaxis": {"title": "Euclidean Distance"}})
    
    if mode == 0:
        fig.update_layout(
            title_text = "Euclidean distance between activations of previous timestep", 
            template="simple_white", 
        )
    elif mode == 1:
        fig.update_layout(
            title_text = "Euclidean distance between gradients of previous timestep",
            template="simple_white", 
        )
        
    fig.update_layout(
        height=500,
        width=700,
    )
    
    fig.update_xaxes(autorange='reversed', showline=True, showgrid=True, linewidth=1, gridcolor="#8a8888",linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, showgrid=True, linecolor='black', gridcolor="#8a8888", mirror=True)
        
    return fig

def line_plot(values, x_range, title, x_title, y_title):
    """
        Line Plot

        Parameters:
        ----------
        values: ndarray
        x_range: 
        title: str
        x_title: str
        y_title: str

        Returns:
        fig: px.line
            Plotly Express line figure.
        --------     
    """

    fig = px.line(x=x_range, y=values, title=title)
    fig.update_layout(
        xaxis_title=x_title, 
        yaxis_title=y_title,
        height=500,
        width=700,
        title_x=0.5,
        template="simple_white", 
        legend=dict(
            title=None, orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    #create border around plot
    fig.update_xaxes(autorange='reversed',showline=True,showgrid=True, linewidth=1,gridcolor="#8a8888", linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1,showgrid=True, linecolor='black',gridcolor="#8a8888", mirror=True)
    
    return fig

def img_plot(values, slider_title, color, title):
    """
        Line Plot
        
        Parameters:
        ----------
        values: ndarray
        slider_title: str
        color: str
        title: str

        Returns:
        fig: px.imshow
            Plotly Express line figure.
        --------     
    """
    
    fig = px.imshow(values, animation_frame=0, labels=dict(animation_frame=slider_title),title=title, zmin=0, zmax=1, color_continuous_scale=color)
   
    fig.update_layout(
        font_color="dark gray",
        yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        coloraxis_colorbar_x=1,
    )
    
     #remove play button
    fig["layout"].pop("updatemenus")
    fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            width=400,
            title_x=0.5,
            font=dict(
                size=11,  # Set the font size here
            )
        )
    

    return fig
                    
def scatter_plot(x, y, title, x_title, y_title):
    """
        
        Parameters:
        ----------
        x:
        y:
        title: str
        x_title: str
        y_title: str

        Returns:
        fig: go.Figure
            Plotly Graph Objects Figure.
        --------     
    """
    
    fig = go.Figure(
            data = [
                go.Scatter(x=x, y=y),
            ],
            layout = {"xaxis": {"title": x_title}, "yaxis": {"title": y_title}, "title": title}
        )

    fig.update_layout(
        height=500,
        width=700,
        title_x=0.5,
        showlegend=False,
        yaxis=dict(showline=True, linewidth=1,showgrid=True, gridcolor="#d3d3d3", linecolor="lightslategray",  mirror=True),
        xaxis=dict(autorange='reversed',showline=True, showgrid=True, linewidth=1, gridcolor="#d3d3d3", linecolor="lightslategray",  mirror=True),
        plot_bgcolor="white",
    )
    return fig

