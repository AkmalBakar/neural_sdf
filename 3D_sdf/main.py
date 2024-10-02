import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
import optax
from models import GridNet3D
from train import train
from plotting import plot_comparison, plot_pos_encoding, plot_feature_grid
from utils import create_opt_vars_helpers_from_filter_spec
from loss import mse_loss
from sdf_functions import *
import matplotlib.pyplot as plt
import igl
from os import path
import numpy as np
import geometry as gm
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing


def collate_fn(batch):
    x_samples, sdf_samples = zip(*batch)
    x_samples = jnp.stack(x_samples)
    sdf_samples = jnp.stack(sdf_samples)
    return x_samples, sdf_samples

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    # Seed random key
    key = jax.random.PRNGKey(0)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    # Load the mesh to approximate
    data_dir = path.join(path.dirname(__file__), "data")
    v, f = igl.read_triangle_mesh(path.join(data_dir, "usb_male.obj"))
    b_min = np.min(v, axis=0)
    b_max = np.max(v, axis=0)
    b_lens = b_max - b_min
    domain_bounds = np.array([b_min - 0.1*b_lens, b_max + 0.1*b_lens])

    # Initialize mesh sampler
    sampler = gm.MeshSampler(v, f, domain_bounds, ratio=0.3, std=0.05*b_lens.min())

    # Data set and loaders
    train_dataset = gm.SDFDataset(sampler, num_samples=1000000)
    test_dataset = gm.SDFDataset(sampler, num_samples=10000)

    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------

    # Initialize network
    model = GridNet3D(
        domain_bounds=domain_bounds,
        num_grid_points=[5, 5, 5],
        feature_size=5,
        width_size=100,
        out_size=1,
        key=key,
    )

    # Define optimization variables
    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda net: net.feature_grid.f, filter_spec, True)
    filter_spec = eqx.tree_at(lambda net: net.mlp, filter_spec, eqx.is_array)

    (
        extract_optimization_variables_from_model,
        combine_optimization_variable_w_model,
        unflatten_opt_vars,
    ) = create_opt_vars_helpers_from_filter_spec(model, filter_spec)

    opt_vars, static_model = extract_optimization_variables_from_model(model)

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------

    # Initialize optimizer
    lr = 1e-3
    opt = optax.adabelief(lr)

    # Run optimization
    train(
        model,
        extract_optimization_variables_from_model,
        combine_optimization_variable_w_model,
        train_loader,
        test_loader,
        mse_loss,
        opt,
        num_epochs=2000,
    )


    # # Plot results
    # grid_pts_x = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 5)
    # grid_pts_y = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 5)

    # # plot_comparison(final_net, f, domain_bounds, grid_pts_x, grid_pts_y, 1000, 1000)
    # # plot_feature_grid(final_net, domain_bounds, grid_pts_x, grid_pts_y, 2000, 2000)
    # # plot_pos_encoding(final_net, domain_bounds, grid_pts_x, grid_pts_y, 2000, 2000)
    # plt.show()
