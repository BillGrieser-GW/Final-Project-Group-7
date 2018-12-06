"""
Utilities for plotting
"""
import torch

def imshowax(ax, img, cmap='Greys_r'):
    """
    Draw either a PIL or Torch image onto the input axes
    """
    #img = img / 2 + 0.5
    if type(img) == torch.Tensor:
        if img.dim() == 3:
            showimg = img.detach().numpy().transpose(1,2,0)
        else:
            showimg = img.detach().numpy()
    else:
        showimg = img
       
    ax.imshow(showimg, cmap=cmap, interpolation='none')
    ax.tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
                   labelbottom=False, labelleft=False)

