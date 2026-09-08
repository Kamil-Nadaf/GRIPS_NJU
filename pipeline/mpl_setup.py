"""Quiet matplotlib when heapy/bayspec request STIX Two Text.

Those packages set ``font.serif = ['STIX Two Text']``. The gbm image has
DejaVu, not STIX Two, so every plot logs ``findfont: Generic family 'serif'...``.
"""

import logging
import warnings

_DONE = False


def silence_missing_fonts():
    """Drop matplotlib font-manager findfont noise (safe to call many times)."""
    global _DONE
    if _DONE:
        return
    _DONE = True
    log = logging.getLogger('matplotlib.font_manager')
    log.setLevel(logging.ERROR)
    log.propagate = False
    warnings.filterwarnings('ignore', message=r'findfont:.*')
    warnings.filterwarnings('ignore', module=r'matplotlib\.font_manager')
