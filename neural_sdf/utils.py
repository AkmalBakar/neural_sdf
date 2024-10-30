from typing import TypeVar

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, Float

# ------------------------------------------------------------------------
# Model splicing utils
# ------------------------------------------------------------------------

# This create functions that help partitions our model into parts we want to optimize and parts that are static
T = TypeVar("T")


def create_opt_vars_helpers_from_filter_spec(model: T, filter_spec):
    opt_vars_tree, _ = eqx.partition(model, filter_spec)
    _, unflatten_opt_vars = jax.flatten_util.ravel_pytree(opt_vars_tree)

    # Creating the function will do the extraction in the future
    def extract_optimization_variables_from_model(
        model: T,
    ) -> tuple[Float[Array, "..."], T]:
        opt_vars_tree, static_model = eqx.partition(model, filter_spec)
        opt_vars, _ = jax.flatten_util.ravel_pytree(opt_vars_tree)
        return opt_vars, static_model

    # Creating the function to combine the static model with the new optimization variables
    def combine_optimization_variables_and_model(
        optimization_variables: Float[Array, "..."], static_model: T
    ) -> T:
        # First we unflatten the optimization variables
        opt_vars_tree = unflatten_opt_vars(optimization_variables)
        # Then we combine it with the static model
        combined_model = eqx.combine(opt_vars_tree, static_model)
        return combined_model

    return (
        extract_optimization_variables_from_model,
        combine_optimization_variables_and_model,
        unflatten_opt_vars,
    )


# This create function that help partitions our model into parts we want to optimize and parts that are static
T = TypeVar("T")


def create_opt_vars_helpers_from_selection_fn(model: T, selection_fn):
    # Doing the optimization variable extraction here once to get the unflatten function
    # Start with a False on all the leaves
    opt_flag_tree = jax.tree_util.tree_map(lambda _: False, model)
    # Set specific leaves to True based on our selection function
    opt_flag_tree = eqx.tree_at(selection_fn, opt_flag_tree, replace_fn=lambda _: True)
    opt_vars_tree, static_tree = eqx.partition(model, opt_flag_tree)
    opt_vars, unflatten_opt_vars = jax.flatten_util.ravel_pytree(opt_vars_tree)

    # Creating the function will do the extraction in the future
    def extract_optimization_variables_from_model(
        model: T,
    ) -> tuple[Float[Array, "..."], T]:
        # Start with a False on all the leaves
        opt_flag_tree = jax.tree_util.tree_map(lambda _: False, model)
        # Set specific leaves to True based on our selection function
        opt_flag_tree = eqx.tree_at(
            selection_fn, opt_flag_tree, replace_fn=lambda _: True
        )
        opt_vars_tree, static_model = eqx.partition(model, opt_flag_tree)
        opt_vars, unflatten_opt_vars = jax.flatten_util.ravel_pytree(opt_vars_tree)
        return opt_vars, static_model

    # Creating the function to combine the static model with the new optimization variables
    def combine_optimization_variables_and_model(
        optimization_variables: Float[Array, "..."], static_model: T
    ) -> T:
        # First we unflatten the optimization variables
        opt_vars_tree = unflatten_opt_vars(optimization_variables)
        # Then we combine it with the static model
        combined_model = eqx.combine(opt_vars_tree, static_model)
        return combined_model

    return (
        extract_optimization_variables_from_model,
        combine_optimization_variables_and_model,
        unflatten_opt_vars,
    )

def compare_pytrees(a, b):
    def compare_leaves(a, b):
        if isinstance(a, jax.Array):
            return jnp.allclose(a, b)
        else:
            return a == b
    return jax.tree_util.tree_all(jax.tree_util.tree_map(compare_leaves, a, b))