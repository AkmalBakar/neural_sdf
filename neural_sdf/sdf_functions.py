import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Float, Array

class Sphere(eqx.Module):
    radius: Float
    dim: int
    bounding_box: Float[Array, "..."]

    def __init__(self, radius, bounding_box, dim = 3):
        self.radius = radius
        self.dim = dim
        self.bounding_box = bounding_box

    def sample_surface(self, num_samples, key):
        ball = jax.random.ball(key, self.dim, 2, (num_samples,))
        sphere = self.radius * ball / jnp.linalg.norm(ball, axis=-1)[:,jnp.newaxis]
        return sphere

    def sample_bounding_box(self, num_samples, key):
        bb_size = self.bounding_box[1] - self.bounding_box[0]
        unit = jax.random.uniform(key, (num_samples, self.dim))
        samples = self.bounding_box[0] + bb_size * unit
        return samples

    def sample(self, num_samples, key, std=0.0, ratio=0.3):
        key1, key2, key3 = jax.random.split(key, 3)
        n_bbox = int(num_samples * ratio)
        n_surf = num_samples - n_bbox

        bbox_samples = self.sample_bounding_box(n_bbox, key=key2)

        # Perturb surface samples
        surf_samples = self.sample_surface(n_surf, key=key1)
        perturb = std * jax.random.normal(key3, surf_samples.shape)
        surf_samples = perturb + surf_samples

        samples = jnp.concat([bbox_samples, surf_samples], axis=0)
        return samples

    def sdf(self, x):
        return jnp.linalg.norm(x, axis=-1) - self.radius