import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
import optax
from models import GridNet2D
from optimization import run_optimization, print_info_opt
from plotting import plot_comparison, plot_pos_encoding, plot_feature_grid
from utils import create_opt_vars_helpers_from_filter_spec
from loss import mse_loss
from sdf_functions import *
import matplotlib.pyplot as plt

# Ground truth SDF
f = circle_sdf

# Domain
domain_bounds = jnp.array([jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0])])

# Seed random key
key = jax.random.PRNGKey(0)

# Initialize network
net = GridNet2D(
    domain_bounds=domain_bounds,
    num_grid_points=[5, 5],
    feature_size=5,
    width_size=100,
    out_size=1,
    key=key,
)

# Define optimization variables
filter_spec = jax.tree.map(lambda _: False, net)
filter_spec = eqx.tree_at(lambda net: net.feature_grid.f, filter_spec, True)
filter_spec = eqx.tree_at(lambda net: net.mlp, filter_spec, eqx.is_array)

(
    extract_optimization_variables_from_model,
    combine_optimization_variable_w_model,
    unflatten_opt_vars,
) = create_opt_vars_helpers_from_filter_spec(net, filter_spec)

opt_vars, static_model = extract_optimization_variables_from_model(net)

# Loss function
def loss_wrt_opt_vars(opt_vars, static_model, x_sample):
    net = combine_optimization_variable_w_model(opt_vars, static_model)
    loss = mse_loss(net, x_sample, f)
    return loss

# Optimization loop
# Subsuming static model
def run_opt(opt_vars_init, x_sample, optimizer, max_iter, tol):
    return run_optimization(loss_wrt_opt_vars, opt_vars_init, static_model, x_sample, optimizer, max_iter, tol)

# Initialize optimizer
opt = optax.chain(print_info_opt(), optax.lbfgs())

# Sample points to evaluate the network
# Reguarly spaced grid
x1s = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 500)
x2s = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 500)
x1_sample, x2_sample = jnp.meshgrid(x1s, x2s)
x_sample_grid = jnp.stack([x1_sample.reshape(-1), x2_sample.reshape(-1)], axis=-1)

# Points on the surface of the shape
key1, key = jax.random.split(key, 2)
x_sample_surface = jax.random.ball(key, 2, 2, (10000,))
# Project to circle surface
x_sample_surface = 3.0 * x_sample_surface / jnp.linalg.norm(x_sample_surface, axis=-1)[:,jnp.newaxis]

x_sample = jnp.concatenate([x_sample_grid, x_sample_surface], axis=0)

# Initial loss
print(f"Initial loss: {loss_wrt_opt_vars(opt_vars, static_model, x_sample)}")

# Run optimization
final_opt_vars, _ = jax.jit(run_opt, static_argnums=(2, 3, 4))(
    opt_vars, x_sample, opt, max_iter=2000, tol=1e-4
)

# Final loss
print(f"Final loss: {loss_wrt_opt_vars(final_opt_vars, static_model, x_sample)}")
final_net = combine_optimization_variable_w_model(final_opt_vars, net)

# Plot results
grid_pts_x = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 5)
grid_pts_y = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 5)

plot_comparison(final_net, f, domain_bounds, grid_pts_x, grid_pts_y, 2000, 2000)
plot_feature_grid(final_net, domain_bounds, grid_pts_x, grid_pts_y, 2000, 2000)
plot_pos_encoding(final_net, domain_bounds, grid_pts_x, grid_pts_y, 2000, 2000)
plt.show()