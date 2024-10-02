import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from typing import NamedTuple
from functools import partial
from loss import mse_loss


@partial(jax.jit, static_argnames=("loss_fn", "optimizer"))
def make_step(opt_vars, x, y, loss_fn, optimizer, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(opt_vars, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_opt_vars = optax.apply_updates(opt_vars, updates)
    return loss, grads, new_opt_vars, opt_state


def train(
    model,
    extract_opt_var_fn,
    combine_opt_var_model_fn,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    num_epochs=5000,
):
    # Split model into opt_vars and static parts
    opt_vars, static_model = extract_opt_var_fn(model)

    # Initialize optimizer
    opt_state = optimizer.init(opt_vars)

    # Redefine loss function in terms of opt_vars
    # Note that we subsume static model here
    def loss_wrt_opt_vars_fn(opt_vars, x_batch, y_batch):
        model = combine_opt_var_model_fn(opt_vars, static_model)
        loss = loss_fn(model, x_batch, y_batch)
        return loss
    # Jitted version to evaluate during validation
    jitted_loss_fn = jax.jit(loss_wrt_opt_vars_fn)

    for epoch in range(num_epochs):

        # Training batches
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch, y_batch = batch
            loss, grads, opt_vars, opt_state = make_step(opt_vars, x_batch, y_batch, loss_wrt_opt_vars_fn, opt_state, optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Final Grads Norm: {jnp.linalg.norm(grads)}")

        # Validation batches
        val_loss = 0.0
        for batch in test_loader:
            x_batch, y_batch = batch
            val_loss += jitted_loss_fn(opt_vars, x_batch, y_batch)
        avg_val_loss = val_loss / len(test_loader)
        print(f"Validation Loss: {avg_val_loss}")

        # Here you could implement logic to resample the dataset if needed
        # if (epoch + 1) % resample_every == 0:
        #     train_loader.dataset.resample()
        #     test_loader.dataset.resample()

    # Finally recombine opt_vars with static_model
    model = combine_opt_var_model_fn(opt_vars, static_model)

    return model