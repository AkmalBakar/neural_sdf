import jax
import jax.numpy as jnp
import equinox as eqx
from interpax import Interpolator3D
from typing import Callable
from jaxtyping import Float, Array
import json

from .datalog import *

# Changing below to an eqx.Module so that compare can be used on it
# Initializing pos encoder
class PosEncoder(eqx.Module):
    domain_bounds: Float[Array, "2 dim"]
    num_pos_encodings: int
    block_size: Float[Array, "dim"]

    def __init__(self, domain_bounds, num_grid_points, num_pos_encodings):
        self.domain_bounds = domain_bounds
        self.num_pos_encodings = num_pos_encodings
        domain_size = self.domain_bounds[1] - self.domain_bounds[0]
        self.block_size = domain_size / (num_grid_points - 1)

    def __call__(self, x):
        domain_pos = x - self.domain_bounds[0]
        block_frac, _ = jnp.modf(domain_pos / self.block_size)
        multipliers = 2 * jnp.pi * jnp.arange(1, self.num_pos_encodings + 1)
        encoding = jnp.sin(jnp.expand_dims(multipliers, -1) * block_frac)
        return encoding

class GridNet3D(eqx.Module):
    feature_grid: Interpolator3D
    pos_encoder: Callable
    mlp: eqx.nn.MLP
    dim: int
    domain_bounds: Float[Array, "2 3"]
    num_grid_points: Float[Array, "3"]
    feature_size: int
    width_size: int
    out_size: int
    num_pos_encodings: int
    activation_fn: Callable

    def __init__(
        self,
        domain_bounds,
        num_grid_points,
        feature_size,
        width_size,
        out_size,
        key,
        num_pos_encodings=3,
        interp_method = "cubic2",
        activation_fn = jax.nn.swish
    ):
        # Store hyperparameters
        self.domain_bounds = jnp.array(domain_bounds)
        self.num_grid_points = jnp.array(num_grid_points)
        self.feature_size = feature_size
        self.width_size = width_size
        self.out_size = out_size
        self.num_pos_encodings = num_pos_encodings
        self.dim = 3
        self.activation_fn = activation_fn

        # Calculating useful dimensions

        # Randomizing keys
        key1, key2 = jax.random.split(key, 2)

        # Initializing feature grid
        grid_pts_x = jnp.linspace(
            self.domain_bounds[0, 0], self.domain_bounds[1, 0], self.num_grid_points[0]
        )
        grid_pts_y = jnp.linspace(
            self.domain_bounds[0, 1], self.domain_bounds[1, 1], self.num_grid_points[1]
        )
        grid_pts_z = jnp.linspace(
            self.domain_bounds[0, 2], self.domain_bounds[1, 2], self.num_grid_points[2]
        )
        feats_shape = (
            grid_pts_x.shape
            + grid_pts_y.shape
            + grid_pts_z.shape
            + tuple([self.feature_size])
        )
        feats_init = jax.random.normal(key1, feats_shape)
        self.feature_grid = Interpolator3D(
            grid_pts_x, grid_pts_y, grid_pts_z, feats_init, method=interp_method, extrap=True
        )

        # # Initializing pos encoder
        # def pos_encoding(x):
        #     domain_pos = x - self.domain_bounds[0]
        #     block_frac, _ = jnp.modf(domain_pos / block_size)
        #     multipliers = 2 * jnp.pi * jnp.arange(1, self.num_pos_encodings + 1)
        #     encoding = jnp.sin(jnp.expand_dims(multipliers, -1) * block_frac)
        #     return encoding

        self.pos_encoder = PosEncoder(self.domain_bounds, self.num_grid_points, self.num_pos_encodings)

        # Initializing MLP
        self.mlp = eqx.nn.MLP(
            self.feature_size + self.num_pos_encodings * self.dim,
            self.out_size,
            self.width_size,
            1,
            self.activation_fn,
            key=key2,
        )

    def get_feature(self, x):
        return self.feature_grid(x[0], x[1], x[2])

    def get_pos_encoding(self, x):
        return self.pos_encoder(x)

    def __call__(self, x):
        feat_vec = self.feature_grid(x[0], x[1], x[2])
        pos_enc = self.pos_encoder(x).reshape(-1)
        out = self.mlp(jnp.concat([feat_vec, pos_enc]))
        return jnp.squeeze(out)

    def save(self, path):
        hyperparameters = {
            "domain_bounds": self.domain_bounds.tolist(),
            "num_grid_points": self.num_grid_points.tolist(),
            "feature_size": self.feature_size,
            "width_size": self.width_size,
            "out_size": self.out_size,
            "num_pos_encodings": self.num_pos_encodings,
            "interp_method": self.feature_grid.method,
            "activation_fn": activation_fn_to_str(self.activation_fn),
        }
        with open(path, "wb") as f:
            hyperparam_str = json.dumps(hyperparameters)
            f.write(f"{hyperparam_str}\n".encode("utf-8"))
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            hyperparam_str = f.readline().decode("utf-8")
            hyperparameters = json.loads(hyperparam_str)
            model = GridNet3D(
                domain_bounds=jnp.array(hyperparameters["domain_bounds"]),
                num_grid_points=jnp.array(hyperparameters["num_grid_points"]),
                feature_size=hyperparameters["feature_size"],
                width_size=hyperparameters["width_size"],
                out_size=hyperparameters["out_size"],
                key=jax.random.PRNGKey(0),
                num_pos_encodings=hyperparameters["num_pos_encodings"],
                interp_method=hyperparameters["interp_method"],
                activation_fn=activation_fn_from_str(hyperparameters["activation_fn"]),
            )
            model = eqx.tree_deserialise_leaves(f, model)
            return model
        
