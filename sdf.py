import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import igl
from os import path
from mayavi import mlab
from interpax import interp3d, Interpolator3D

def draw_boxes(box_corners, fig, color=(1,1,1), line_width=1, color_list=None):
    ''' Draw 3D axis-aligned boxes
    Args:
        box_corners: numpy array (n,2,3) for min and max XYZ of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(box_corners)
    for n in range(num):
        b_min = box_corners[n, 0, :]
        b_max = box_corners[n, 1, :]
        if color_list is not None:
            color = color_list[n] 
        
        # Generate 8 corners of the box
        corners = np.array([
            [b_min[0], b_min[1], b_min[2]],
            [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]],
            [b_min[0], b_max[1], b_max[2]]
        ])
        
        # if draw_text: 
        #     center = (b_min + b_max) / 2
        #     mlab.text3d(center[0], center[1], center[2], '%d'%n, scale=text_scale, color=color, figure=fig)
        
        # Draw the box edges
        for k in range(4):
            i,j = k, (k+1)%4
            mlab.plot3d([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], [corners[i,2], corners[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
            mlab.plot3d([corners[i+4,0], corners[j+4,0]], [corners[i+4,1], corners[j+4,1]], [corners[i+4,2], corners[j+4,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
            mlab.plot3d([corners[i,0], corners[i+4,0]], [corners[i,1], corners[i+4,1]], [corners[i,2], corners[i+4,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    return fig

# Read mesh
v, f = igl.read_triangle_mesh(path.join(path.dirname(__file__), "usb_male.obj"))

# Get mesh bounding box
bb_min = v.min(axis=0)
bb_max = v.max(axis=0)

# Add margin around the tight bounding box
largest_side = (bb_max - bb_min).max()
margin_frac = 0.05
bb_min = bb_min - margin_frac * largest_side
bb_max = bb_max + margin_frac * largest_side
bb_corners = np.array([bb_min, bb_max])

xyzs = jnp.linspace(bb_min, bb_max, )
Interpolator3D()
# Define the architecture
# class SDFNet(eqx.Module):
#     feature_grid: eqx.Module
#     mlp: eqx.Module
    
#     def __init__()

# Plot mesh and bb
fig = mlab.figure(1, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.1,0.1,0.1))
mp = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f, color=(0.5,0.5,0.0))
draw_boxes(np.expand_dims(bb_corners, axis=0), fig=fig, color=(0,0,0))
mlab.show()

# Create a dataloader for location and signed distaance

print()
