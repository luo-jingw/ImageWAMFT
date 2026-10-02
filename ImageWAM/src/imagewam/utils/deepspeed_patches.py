"""Runtime patches for the pinned DeepSpeed (0.18.5); they change speed only, never the computed values."""

import torch

from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


def cache_used_parameter_count() -> bool:
    """Compute DeepSpeed's per-backward used-parameter count once per autograd graph task.

    ZeRO-1/2 gradient hooks call `count_used_parameters_in_backward(all_params)` once per parameter, which walks
    every parameter (`view_as` + node lookup) and makes each backward O(n_params^2) on the CPU: with the 594
    trainable tensors of the Qwen-Image-2.1 stack that is ~353k `view_as` calls and ~1 s of CPU per micro-step.
    The count only depends on the running graph task (its execution info is fixed when the task starts), so the
    cached value equals what every call would return.
    """
    try:
        from deepspeed.runtime.zero import stage_1_and_2
    except ImportError:
        return False
    original = getattr(stage_1_and_2, "count_used_parameters_in_backward", None)
    if original is None:
        return False
    if getattr(original, "_imagewam_cached", False):
        return True
    cache: dict[tuple[int, int], int] = {}

    def cached(parameters):
        key = (torch._C._current_graph_task_id(), id(parameters))
        value = cache.get(key)
        if value is None:
            cache.clear()
            value = original(parameters)
            cache[key] = value
        return value

    cached._imagewam_cached = True
    stage_1_and_2.count_used_parameters_in_backward = cached
    return True
