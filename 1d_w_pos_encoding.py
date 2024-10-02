import jax.numpy as jnp
import jax
import equinox as eqx
from interpax import *
import matplotlib.pyplot as plt
from utils import *
from typing import Callable, NamedTuple
import optax
from optax import tree_utils as otu

# Ground truth function
# f = lambda x: jnp.sin(x) + jnp.cos(2 * x + 1.0) + 0.2 * jnp.sin(10 * x + 2)
# f = lambda x: jnp.pow(x, 2)
f = lambda x: jnp.select(
    [x < i for i in range(1, 11)], [1, 5, 2, 8, 3, 7, 4, 9, 6, 0], default=0
)


# Initialize interpolation

# Random features at grid points
key = jax.random.PRNGKey(0)
key1, key = jax.random.split(key, 2)

# ---------------------------------------------------------------------------
# Define network
# ---------------------------------------------------------------------------


class PiecewiseNet(eqx.Module):
    feature_grid: Interpolator1D
    pos_encoder: Callable
    mlp: eqx.nn.MLP

    def __init__(self, domain_bounds, num_grid_points,
                 feature_size, width_size, out_size, key):

        domain_size = domain_bounds[1] - domain_bounds[0]
        # Distance between grid points
        block_size = domain_size / (num_grid_points - 1)

        num_pos_encodings = 3
        def pos_encoding(x):
            # This encodes the position between grid points
            domain_pos = x - domain_bounds[0]
            block_frac, _ = jnp.modf(domain_pos / block_size)
            multipliers = 2 * jnp.pi * jnp.arange(1, num_pos_encodings + 1)
            encoding = jnp.sin(multipliers * block_frac)
            return encoding

        key1, key2 = jax.random.split(key, 2)

        grid_points = jnp.linspace(domain_bounds[0], domain_bounds[1], num_grid_points)
        feats_init = jax.random.normal(key1, grid_points.shape + tuple([feature_size]))
        self.feature_grid = Interpolator1D(grid_points, feats_init, method="cubic2")
        self.pos_encoder = pos_encoding
        self.mlp = eqx.nn.MLP(
            feature_size + num_pos_encodings, out_size, width_size, 1, jax.nn.swish, key=key2
        )

    def __call__(self, x):
        feat_vec = self.feature_grid(x)
        pos_enc = self.pos_encoder(x)
        out = self.mlp(jnp.concat([feat_vec, pos_enc]))
        return out


net = PiecewiseNet(
    domain_bounds=[0.0, 10.0],
    num_grid_points=5,
    feature_size=5,
    width_size=100,
    out_size=1,
    key=key1,
)

net(1.0)
net(3.0)
net(6.0)


grid_points = jnp.linspace(0.0, 10.0, 5)

# initial interpolation vals
plot_xs = jnp.linspace(0.0, 10.0, 1000)
initial_interp_vals = vmap(net)(plot_xs)
# plt.plot(plot_xs, f(plot_xs), label="Ground truth")
# plt.plot(plot_xs, vmap(net)(plot_xs), label="Interpolation")
# plt.vlines(grid_points, -3, 3)
# plt.legend()
# plt.show()

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


# loss function
def loss_fn(net, x_sample):
    f_pred = vmap(net)(x_sample)
    f_true = f(x_sample)
    loss_per_sample = vmap(lambda p, t: jnp.dot(p - t, p - t))(f_pred, f_true)
    return jnp.average(loss_per_sample)


# loss function wrt optimization vars
def loss_wrt_opt_vars(opt_vars, static_model, x_sample):
    net = combine_optimization_variable_w_model(opt_vars, static_model)
    return loss_fn(net, x_sample)


# Defining optimizing loop
def run_lbfgs(loss_fn, opt_vars_init, static_model, x_sample, opt, max_iter, tol):
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


# Defining optimizer
opt = optax.chain(print_info(), optax.lbfgs())

# Sampled points
x_sample = jnp.linspace(0.0, 10.0, 2000)

print(f"Initial loss: {loss_wrt_opt_vars(opt_vars, static_model, x_sample)}")

final_opt_vars, _ = run_lbfgs(
    loss_wrt_opt_vars, opt_vars, static_model, x_sample, opt, max_iter=2000, tol=1e-4
)

print(f"Final loss: {loss_wrt_opt_vars(final_opt_vars, static_model, x_sample)}")
final_net = combine_optimization_variable_w_model(final_opt_vars, net)

# Plot final comparison
plot_xs = jnp.linspace(0.0, 10.0, 1000)
final_interp_vals = vmap(final_net)(plot_xs)
plt.plot(plot_xs, f(plot_xs), label="Ground truth")
plt.plot(plot_xs, initial_interp_vals, label="Initial Interpolation")
plt.plot(plot_xs, final_interp_vals, label="Final Interpolation")
plt.legend()
plt.show(block=True)
