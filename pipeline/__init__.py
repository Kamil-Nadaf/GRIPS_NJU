"""Modular Fermi GBM preprocessing + spectral-fitting pipeline.

Import the runner for custom projects::

    from pipeline.runner import GRBPipelineRunner
    from pipeline.context import GRBContext

    runner = GRBPipelineRunner(model_name='cpl', include_bgo=False)
    ctx = runner.run('GRB140606B')  # tint-first default stages
"""

from .mpl_setup import silence_missing_fonts

silence_missing_fonts()
