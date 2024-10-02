import jax.numpy as jnp
from jax import vmap

def mse_loss(net, x_sample, true_func):
    """
    Compute the Mean Squared Error loss between the network predictions and the true function.
    
    Args:
    net: The neural network model
    x_sample: Input samples
    true_func: The true function to compare against
    
    Returns:
    The average MSE loss
    """
    def sqr_err_per_sample(x_sample):
        err = net(x_sample) - true_func(x_sample)
        return jnp.dot(err, err)
    loss_per_sample = vmap(sqr_err_per_sample)(x_sample)
    return jnp.average(loss_per_sample)