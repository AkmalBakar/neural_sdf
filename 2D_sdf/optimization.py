import jax
import optax
from optax import tree_utils as otu
from typing import NamedTuple

# Optimization loop
# Wrap this function to subsume the static variables before jitting
def run_optimization(loss_fn, opt_vars_init, static_model, x_sample, opt, max_iter, tol):
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


def print_info_opt():
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
