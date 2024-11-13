from datetime import datetime
from importlib import resources
import json
import os
from os import path
from typing import *

import equinox as eqx
from jax import nn, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
from torch.utils.tensorboard import SummaryWriter
import yaml

from .plot import *

# ------------------------------------------------------------------------
# Book-keeping utils
# ------------------------------------------------------------------------

def add_time_stamp(in_str: str):
    base_str, ext = path.splitext(in_str)
    return f"{base_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

def load_config(config_path: str, default_config_path: str = None) -> Dict:
    if default_config_path is None:
        default_config_path = get_default_config()

    # Load default configuration
    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)
    
    # Load user configuration
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    # Recursively update default config with user config
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    # Merge configurations
    final_config = update_dict(default_config, user_config)
    
    return final_config

def default_config_path():
    """Returns the path to the default configuration file."""
    with resources.path('neural_sdf.configs', 'default.yaml') as config_path:
        return str(config_path)

def get_default_config():
    """Returns the path to the default configuration file."""
    with resources.path('neural_sdf.configs', 'default.yaml') as config_path:
        return str(config_path)

def complete_config(user_config: str, default_config_path: Optional[str] = None) -> Dict:
    if default_config_path is None:
        default_config_path = get_default_config()

    # Load default configuration
    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)
    
    # Recursively update default config with user config
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    # Merge configurations
    final_config = update_dict(default_config, user_config)
    
    return final_config
    
fn_map = {"swish": nn.swish, "relu": nn.relu}
def activation_fn_from_str(func_str: str):
    if func_str not in fn_map.keys():
        raise ValueError(f"{func_str} not in list of activation funcs available. Select from {fn_map.keys()}")
    return fn_map[func_str]
    
def activation_fn_to_str(func: Callable):
    # Flip the keys and values
    fn_map_inv = {v: k for k, v in fn_map.items()}
    return fn_map_inv[func]
    

def save_config(config, run_dir):
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

# def save_model(model, out_dir, model_name, epoch):
#     os.makedirs(out_dir, exist_ok=True)
#     filename = os.path.join(out_dir, f"{model_name}_epoch_{epoch}.eqx")
#     with open(filename, "wb") as f:
#         eqx.tree_serialise_leaves(f, model)
#     return filename

# ------------------------------------------------------------------------
# Logging utils
# ------------------------------------------------------------------------

def create_summary_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def print_metrics(metrics: dict, epoch):
    print(f"Epoch {epoch}:")
    for key, value in metrics.items():
        print(f"    {key}: {np.asarray(value)}")

def log_metrics(writer: SummaryWriter, metrics: dict, epoch):
    for key, value in metrics.items():
        writer.add_scalar(key, np.asarray(value), epoch)


def write_slice_comparison(
    model: callable,
    mesh_sdf_func: callable,
    writer: SummaryWriter,
    epoch: int,
    domain_bounds: Float[Array, "2 3"],
    slice_axis: int = 2,
    location: float = 0.0,
    n_pts=100,
):
    # Turn off interative figure
    mpl_backend = mpl.get_backend()
    mpl.use("Agg")

    fig = plot_comparison_sdf_slice(
        model, mesh_sdf_func, domain_bounds, slice_axis, location, n_pts
    )

    writer.add_figure(f"SDF_slice/axis_{slice_axis}={location}", fig, epoch)
    plt.close(fig)

    # Reset backend
    mpl.use(mpl_backend)
    return