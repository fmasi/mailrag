"""Cheap, LLM-free chunk-size calibration from cleaned body token lengths.

ZTH onboarding profiles chunk_size per-corpus (body-length distributions vary
widely across mailboxes); this is the one structural param it calculates.
Factored out of scripts/build_local_eml_rag.py so the build script and the
onboard flow share one implementation.
"""

import math


def percentiles(values, ps=(50, 90, 95, 99, 100)):
    """Nearest-rank percentiles (matches the historical build-script behaviour)."""
    s = sorted(values)
    return {p: s[min(len(s) - 1, int(len(s) * p / 100))] for p in ps}


def suggest_chunk_size(token_lengths, default=512):
    """p90 of the (positive) token lengths, rounded up to the nearest 64 and
    clamped to [256, 1024]. Empty/degenerate input -> ``default``."""
    lengths = [n for n in token_lengths if n > 0]
    if not lengths:
        return default
    p90 = percentiles(lengths)[90]
    return min(1024, max(256, int(math.ceil(p90 / 64) * 64)))
