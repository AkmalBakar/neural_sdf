import jax.numpy as jnp
from jax import vmap

def mse_loss(model, x_samples, y_samples):
    """
    Compute the Mean Squared Error loss between the network predictions and the true function.
    
    Args:
    net: The neural network model
    x_samples: Input samples
    y_samples: The true function values
    
    Returns:
    The average MSE loss
    """
    def sqr_err_per_sample(x, y):
        err = model(x) - y
        return jnp.dot(err, err)
    loss_per_sample = vmap(sqr_err_per_sample, in_axes=(0,0))(x_samples, y_samples)
    return jnp.average(loss_per_sample)