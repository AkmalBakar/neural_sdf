import jax.numpy as jnp
from jax import vmap, grad, value_and_grad, lax

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

def mse_loss_with_max(y_preds, y_truths):
    """
    Compute the Mean Squared Error loss between the network predictions and the true function.
    Additionally returns the maximum error.
    
    Args:
    net: The neural network model
    y_preds: The predicted function values
    y_truths: The true function values
    
    Returns:
    The average MSE loss, maximum error
    """
    def sqr_err_per_sample(y_pred, y_truth):
        err = y_pred - y_truth
        return jnp.dot(err, err)
    loss_per_sample = vmap(sqr_err_per_sample, in_axes=(0,0))(y_preds, y_truths)
    max_error = jnp.sqrt(loss_per_sample.max())
    return jnp.average(loss_per_sample), max_error

def mse_loss_with_max_and_frac(y_preds, y_truths):
    """
    Compute the Mean Squared Error loss between the network predictions and the true function.
    Additionally returns the maximum error and the maximum error percentage.
    
    Args:
    net: The neural network model
    y_preds: The predicted function values
    y_truths: The true function values
    
    Returns:
    The average MSE loss, maximum error
    """
    def sqr_err_and_percentage_per_sample(y_pred, y_truth):
        err = y_pred - y_truth
        err_perc = jnp.abs(err) / (jnp.abs(y_truth)+1e-8)
        return jnp.dot(err, err), err_perc
    loss_per_sample, err_perc_per_sample = vmap(sqr_err_and_percentage_per_sample, in_axes=(0,0))(y_preds, y_truths)
    max_error = jnp.sqrt(loss_per_sample.max())
    max_error_percentage = err_perc_per_sample.max()
    return jnp.average(loss_per_sample), max_error, max_error_percentage


def closest_point_loss_with_max(model, x_samples, pred_sdf_vals, grad_xs, domain_bounds):
    """
    Measures how far from the zero level set a point is mapped by the closest point function.
    Note: If closest point for x falls out of domain bounds, we will ignore x.
    
    Args:
    model: The neural network model
    x_samples: Input samples
    pred_sdf_vals: Predicted SDF values
    grad_xs: Gradient of model at x_samples
    
    Returns:
    Average closest point error, maximum error
    """
    def in_domain(x):
        return jnp.all(jnp.logical_and(x >= domain_bounds[0], x <= domain_bounds[1]))
    
    def closest_point_error_per_sample(x, sdf_val, grad_x):
        cp = x - sdf_val * grad_x
        s = model(x - sdf_val * grad_x)
        s_sqr = jnp.square(s) 
        # Set error to zero for closest points outside domain
        res = lax.select(in_domain(x), s_sqr, 0.0)
        return res 
    loss_per_sample = vmap(closest_point_error_per_sample, in_axes=(0, 0, 0))(
        x_samples, pred_sdf_vals, grad_xs
    )
    max_error = jnp.sqrt(loss_per_sample.max())
    return jnp.average(loss_per_sample), max_error

def eikonal_loss_with_max(grad_xs):
    """
    Measure the deviation of the gradient from unit vector.
    Signed distance fields must have gradients that are unit vectors everywhere.
    
    Args:
    grads: Gradient of SDF model at each sample point
    
    Returns:
    Average eikonal loss, maximum error
    """
    def eikonal_loss_per_sample(grad):
        deviation = jnp.linalg.norm(grad) - 1
        dev_sqr = jnp.square(deviation)
        return dev_sqr
    loss_per_sample = vmap(eikonal_loss_per_sample)(grad_xs)
    max_error = jnp.sqrt(loss_per_sample.max())
    return jnp.average(loss_per_sample), max_error
