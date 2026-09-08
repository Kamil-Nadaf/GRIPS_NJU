"""bayspec Plot.infer wrappers."""

from pipeline.context import get_current
from pipeline.fitting import plot_infer_fit as _plot_infer_fit


def plot_infer_fit(model_name='cpl', mode='tintegrated', slice_index=None,
                   style='CE', fixed_params=None, nlive=1000, savepath=None,
                   display=True, ctx=None, include_bgo=False):
    ctx = ctx or get_current()
    return _plot_infer_fit(
        ctx, model_name=model_name, mode=mode, slice_index=slice_index,
        style=style, fixed_params=fixed_params, nlive=nlive,
        savepath=savepath, display=display, include_bgo=include_bgo)
