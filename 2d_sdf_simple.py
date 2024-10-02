import jax.numpy as jnp
import jax
import equinox as eqx
from interpax import *
import matplotlib.pyplot as plt
from utils import *
from typing import Callable, NamedTuple
import optax
from optax import tree_utils as otu

# jax config flags
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_log_compiles", True)

# Ground truth SDF
# Circle of radius 3
f = lambda x : jnp.linalg.norm(x) - 3

# Domain
domain_bounds = jnp.array([jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0])])

# Seed random key
key = jax.random.PRNGKey(0)

# ---------------------------------------------------------------------------
# Define network
# ---------------------------------------------------------------------------


class GridNet(eqx.Module):
    feature_grid: Interpolator2D
    pos_encoder: Callable
    mlp: eqx.nn.MLP
    dim: int = 2

    def __init__(self, domain_bounds, num_grid_points,
                 feature_size, width_size, out_size, key):

        # Ensure in array form
        domain_bounds = jnp.array(domain_bounds)
        num_grid_points = jnp.array(num_grid_points)

        # Calculating useful dimensions
        domain_size = domain_bounds[1] - domain_bounds[0]
        # Distance between grid points
        block_size = domain_size / (num_grid_points - 1)

        # Randomizing keys
        key1, key2 = jax.random.split(key, 2)

        # Initializing feature grid
        grid_pts_x = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], num_grid_points[0])
        grid_pts_y = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], num_grid_points[1])
        feats_shape = grid_pts_x.shape + grid_pts_y.shape + tuple([feature_size])
        feats_init = jax.random.normal(key1, feats_shape)
        self.feature_grid = Interpolator2D(grid_pts_x, grid_pts_y, feats_init, method="cubic2")

        # Initializing pos encoder
        num_pos_encodings = 3
        def pos_encoding(x):
            # This encodes the position between grid points
            domain_pos = x - domain_bounds[0]
            block_frac, _ = jnp.modf(domain_pos / block_size)
            multipliers = 2 * jnp.pi * jnp.arange(1, num_pos_encodings + 1)
            encoding = jnp.sin(jnp.expand_dims(multipliers, -1) * block_frac)
            return encoding
        self.pos_encoder = pos_encoding

        # Initializing MLP
        self.mlp = eqx.nn.MLP(
            feature_size + num_pos_encodings * self.dim, out_size, width_size, 1, jax.nn.swish, key=key2
        )

    def get_feature(self, x):
        return self.feature_grid(x[0], x[1])

    def get_pos_encoding(self, x):
        return self.pos_encoder(x)

    def __call__(self, x):
        feat_vec = self.feature_grid(x[0], x[1])
        pos_enc = self.pos_encoder(x).reshape(-1)
        out = self.mlp(jnp.concat([feat_vec, pos_enc]))
        return out


net = GridNet(
    domain_bounds=domain_bounds,
    num_grid_points=[5,5],
    feature_size=5,
    width_size=100,
    out_size=1,
    key=key,
)

# ---------------------------------------------------------------------------
# Plotting partial outputs of the initial network
# ---------------------------------------------------------------------------

# grid_pts_x = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 5)
# grid_pts_y = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 5)


# # Plotting pos encoding
# xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 1000)
# ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 1000)
# plot_xs, plot_ys = jnp.meshgrid(xs, ys)
# plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
# feat_interp = vmap(vmap(net.get_pos_encoding))(plot_pts)

# plt.figure(1)
# fig, axs = plt.subplots(3, 2, figsize=(12, 18))
# fig.suptitle('Pos Encoding plots')

# for i in range(3):
#     for j in range(2):
#         ax = axs[i, j]
#         im = ax.pcolormesh(plot_xs, plot_ys, feat_interp[:, :, i, j], shading='auto')
#         ax.set_title(f'Subplot {i+1},{j+1}')
#         ax.set_xlabel('X')
#         ax.set_ylabel("Y")
#         fig.colorbar(im, ax=ax)

#         # Add vertical lines
#         for x in grid_pts_x:
#             ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

#         # Add horizontal lines
#         for y in grid_pts_y:
#             ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# # plt.show()

# # Plotting Feature Grid
# xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 1000)
# ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 1000)
# plot_xs, plot_ys = jnp.meshgrid(xs, ys)
# plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
# feat_interp = vmap(vmap(net.get_feature))(plot_pts)

# plt.figure(2)
# fig, axs = plt.subplots(1, 5, figsize=(20, 10))
# fig.suptitle('Feature Grid Plot')

# for i in range(5):
#     ax = axs[i]
#     im = ax.pcolormesh(plot_xs, plot_ys, feat_interp[:, :, i], shading='auto')
#     ax.set_title(f'Subplot {i+1}')
#     ax.set_xlabel('X')
#     ax.set_ylabel("Y")
#     fig.colorbar(im, ax=ax)

#     # Add vertical lines
#     for x in grid_pts_x:
#         ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

#     # Add horizontal lines
#     for y in grid_pts_y:
#         ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.show(block=True)

# ---------------------------------------------------------------------------
# Defining optimization variables
# ---------------------------------------------------------------------------

# Define optimization variables
# First unselect all
filter_spec = jax.tree.map(lambda _: False, net)
# Select feature grid features
filter_spec = eqx.tree_at(lambda net: net.feature_grid.f, filter_spec, True)
# Select all arrays in MLP
filter_spec = eqx.tree_at(lambda net: net.mlp, filter_spec, eqx.is_array)

(
    extract_optimization_variables_from_model,
    combine_optimization_variable_w_model,
    unflatten_opt_vars,
) = create_opt_vars_helpers_from_filter_spec(net, filter_spec)

opt_vars, static_model = extract_optimization_variables_from_model(net)


# ---------------------------------------------------------------------------
# Defining loss function
# ---------------------------------------------------------------------------

# loss function
def loss_fn(net, x_sample):
    f_pred = vmap(net)(x_sample)
    f_true = vmap(f)(x_sample)
    loss_per_sample = vmap(lambda p, t: jnp.dot(p - t, p - t))(f_pred, f_true)
    return jnp.average(loss_per_sample)


# loss function wrt optimization vars
def loss_wrt_opt_vars(opt_vars, static_model, x_sample):
    net = combine_optimization_variable_w_model(opt_vars, static_model)
    return loss_fn(net, x_sample)


# ---------------------------------------------------------------------------
# Defining optimization
# ---------------------------------------------------------------------------

# Defining optimizing loop
def run_optimization(loss_fn, opt_vars_init, x_sample, opt, max_iter, tol):
    # First let's subsume static variables into the loss func
    def loss_func(opt_vars):
        return loss_fn(opt_vars, static_model, x_sample)

    value_and_grad_fun = optax.value_and_grad_from_state(loss_func)

    def step(carry):
        opt_vars, state = carry
        loss, grad = value_and_grad_fun(opt_vars, state=state)
        updates, state = opt.update(
            grad, state, opt_vars, value=loss, grad=grad, value_fn=loss_func
        )
        opt_vars = optax.apply_updates(opt_vars, updates)
        return opt_vars, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (opt_vars_init, opt.init(opt_vars_init))
    final_opt_vars, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_opt_vars, final_state


class InfoState(NamedTuple):
    iter_num: int


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Value: {v}, Gradient norm: {e}",
            i=state.iter_num,
            v=value,
            e=otu.tree_l2_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

# ---------------------------------------------------------------------------
# Running optimization
# ---------------------------------------------------------------------------

# Initializing optimizer
opt = optax.chain(print_info(), optax.lbfgs())

# Sample the interpolation
x1s = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 1000)
x2s = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 1000)
x1_sample, x2_sample = jnp.meshgrid(x1s, x2s)
x_sample = jnp.stack([x1_sample.reshape(-1), x2_sample.reshape(-1)], axis=-1)


print(f"Initial loss: {loss_wrt_opt_vars(opt_vars, static_model, x_sample)}")

final_opt_vars, _ = jax.jit(run_optimization, static_argnums=(0,3,4,5))(
    loss_wrt_opt_vars, opt_vars, x_sample, opt, max_iter=2000, tol=1e-4
)

print(f"Final loss: {loss_wrt_opt_vars(final_opt_vars, static_model, x_sample)}")
final_net = combine_optimization_variable_w_model(final_opt_vars, net)

# Plot final comparison
xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 2000)
ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 2000)
plot_xs, plot_ys = jnp.meshgrid(xs, ys)
plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
final_pred = vmap(vmap(final_net))(plot_pts).squeeze()
final_true = vmap(vmap(f))(plot_pts).squeeze()


plt.figure(4)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Final Comparison')

data = [final_pred, final_true]
label = ["Final Prediction", "Ground Truth"]

grid_pts_x = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], 5)
grid_pts_y = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], 5)

for i in range(2):
    ax = axs[i]
    # im = ax.pcolormesh(plot_xs, plot_ys, data[i], shading='auto')
    im = ax.contour(plot_xs, plot_ys, data[i], levels=[-0.5,0.0,0.5])
    ax.set_title(f'{label[i]}')
    ax.set_xlabel('X')
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax)

    # Add vertical lines
    for x in grid_pts_x:
        ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

    # Add horizontal lines
    for y in grid_pts_y:
        ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show(block=True)