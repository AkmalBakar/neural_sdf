import jax
import jax.numpy as jnp
import equinox as eqx
from interpax import Interpolator3D
from typing import Callable

class GridNet3D(eqx.Module):
    feature_grid: Interpolator3D
    pos_encoder: Callable
    mlp: eqx.nn.MLP
    dim: int = 3

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
        grid_pts_z = jnp.linspace(domain_bounds[0,2], domain_bounds[1,2], num_grid_points[2])
        feats_shape = grid_pts_x.shape + grid_pts_y.shape + grid_pts_x.shape + tuple([feature_size])
        feats_init = jax.random.normal(key1, feats_shape)
        self.feature_grid = Interpolator3D(grid_pts_x, grid_pts_y, grid_pts_z, feats_init, method="cubic2")

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
        feat_vec = self.feature_grid(x[0], x[1], x[2])
        pos_enc = self.pos_encoder(x).reshape(-1)
        out = self.mlp(jnp.concat([feat_vec, pos_enc]))
        return out
