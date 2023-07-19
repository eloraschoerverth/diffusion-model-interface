import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation

#https://jinjeon.me/post/mutual-info/
def plot_hist1d(sample, bins=20):
    """
    one dimensional histogram of the slices

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    histogram
        comparing two images side by side
    """
    i = 0
    for s in sample:
        fig, axes = plt.subplots(ncols=2, figsize=(10,4)) 
        axes[0].set_ylim([0, 600])
        axes[0].set_xlim([-3, 3])
        axes[0].hist(s.ravel(), bins)
        axes[0].set_title('Hist'+str(i*100))
        axes[1].imshow(s[:,:,0], cmap="gray") 
        axes[1].set_title('Step'+str(900+i*10))
        axes[1].axis('off')
        plt.savefig('celeb_histo/mu'+str(i).zfill(3)+'.png')
        i+=1
 

    plt.show()




def plot_scatter2d(img1, img2):
    """
    plot the two image's histogram against each other

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    Returns
    -------
    2d plotting of the two images and correlation coeeficient
    """
    corr = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    plt.plot(img1.ravel(), img2.ravel(), '.')

    plt.xlabel('Img1 signal')
    plt.ylabel('Img2 signal')
    plt.title('Img1 vs Img2 signal cc=' + str(corr))
    plt.show()

def plot_joint_histogram(img1, img2, i, bins=20, log=True):
    """
    plot sample, histogram and joint histogram w/ first sample
    
    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram
    log: boolean (default=True)
        keeping it true will show a better contrasted image

    Returns
    -------
    joint histogram
        feature space of the two images in graph

    """
    fig, axes = plt.subplots(ncols=3, figsize=(10,4)) 
    
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)
    # transpose to put the T1 bins on the horizontal axis and use 'lower' to put 0, 0 at the bottom of the plot
    if not log:
        axes[0].imshow(hist_2d.T, origin='lower')
        axes[0].xlabel('Img1 signal bin')
        axes[0].ylabel('Img2 signal bin')
        

    # log the values to reduce the bins with large values
    hist_2d_log = np.zeros(hist_2d.shape)
    non_zeros = hist_2d != 0
    hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
    axes[0].imshow(s[:,:,0], cmap="gray") 
    axes[0].set_title('Sample at '+str(i*100))
    axes[0].axis('off')
    #axes[1].set_ylim([0, 600])
    axes[1].set_xlim([-3, 3])
    axes[1].hist(s.ravel(), bins)
    axes[1].set_title('Hist'+str(i*100))
    axes[2].imshow(hist_2d_log.T, origin='lower')
    axes[2].set_title('Joint Histogram')
    
    
    plt.savefig('histogram/2d'+str(i)+'.png')
    plt.show()
    
    
def mutual_information(img1, img2, bins=20):
    """
    measure the mutual information of the given two images

    Parameters
    ----------
    img1: numpy array

    img2: numpy array

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float

    """
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def entropy(img):
    return None



