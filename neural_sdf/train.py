import jax
import jax.numpy as jnp
import optax
from functools import partial
from .loss import *
from jax import vmap
import jax
import time

@partial(jax.jit, static_argnames=("loss_fn", "optimizer"))
def make_step(opt_vars, x, y, loss_fn, optimizer, opt_state):
    (loss,aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_vars, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_opt_vars = optax.apply_updates(opt_vars, updates)
    return loss, grads, new_opt_vars, opt_state, aux_metrics


def train(
    model,
    extract_opt_var_fn,
    combine_opt_var_model_fn,
    train_loader,
    test_loader,
    loss_fn,
    loss_aux_metrics,
    optimizer,
    num_epochs=5000,
    write_fn=None,
    save_fn=None,
    save_interval=100,
    plot_fn=None,
    plot_interval=100
):
    # Split model into opt_vars and static parts
    opt_vars, static_model = extract_opt_var_fn(model)

    # Initialize optimizer
    opt_state = optimizer.init(opt_vars)

    # Redefine loss function in terms of opt_vars
    # Note that we subsume static model here
    def loss_wrt_opt_vars_fn(opt_vars, x_batch, y_batch):
        model = combine_opt_var_model_fn(opt_vars, static_model)
        loss, max_error = loss_fn(model, x_batch, y_batch)
        return loss, max_error
    # Jitted version to evaluate during validation
    jitted_loss_fn = jax.jit(loss_wrt_opt_vars_fn)

    # Initialize path to saved model
    latest_model_path = None
    start_time = time.perf_counter()

    for epoch in range(num_epochs):

        # Training batches
        epoch_loss = 0.0
        train_aux_metrics = loss_aux_metrics
        for i,batch in enumerate(train_loader):
            x_batch, y_batch = batch
            loss, grads, opt_vars, opt_state, train_batch_aux_metrics = make_step(opt_vars, x_batch, y_batch, loss_wrt_opt_vars_fn, optimizer, opt_state)
            epoch_loss += loss

            # Assuming that all aux_metric are max ops
            train_aux_metrics = jax.tree.map(lambda a,b: jnp.maximum(a,b), train_aux_metrics, train_batch_aux_metrics)

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation batches
        val_loss = 0.0
        test_aux_metrics = loss_aux_metrics
        for batch in test_loader:
            x_batch, y_batch = batch

            val_batch_loss, test_batch_aux_metrics = jitted_loss_fn(opt_vars, x_batch, y_batch)
            val_loss += val_batch_loss

            # Assuming that all aux_metric are max ops
            test_aux_metrics = jax.tree.map(lambda a,b: jnp.maximum(a,b), test_aux_metrics, test_batch_aux_metrics)

        avg_val_loss = val_loss / len(test_loader)

        # Get updated model before plotting and saving
        model = combine_opt_var_model_fn(opt_vars, static_model)

        # Write out metric
        metrics = {
            "Loss/train": avg_train_loss,
            "Loss/validation": avg_val_loss,
            "Gradients/train_grad_norm": jnp.linalg.norm(grads),
            "Elapsed time (s)": time.perf_counter() - start_time,
        }
        for key, metric in train_aux_metrics.items():
            metrics["Train " + key] = metric
        for key, metric in test_aux_metrics.items():
            metrics["Test " + key] = metric

        if write_fn is not None:
            write_fn(metrics, epoch + 1)

        # Save model
        if (epoch + 1) % save_interval == 0 and save_fn is not None:
            latest_model_path = save_fn(model, epoch + 1)

        # Save plot
        if (epoch + 1) % plot_interval == 0 and plot_fn is not None:
            plot_fn(model, epoch + 1)

    # Finally recombine opt_vars with static_model
    model = combine_opt_var_model_fn(opt_vars, static_model)

    # Save final model
    if save_fn is not None:
        latest_model_path = save_fn(model, epoch + 1)

    return model, latest_model_path 
