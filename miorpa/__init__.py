"""MIORPA 2026 - pluralistic activation steering for small language models.

Typical use from the driver notebook:

    from miorpa import config as cfg
    from miorpa import run_pipeline as rp

    run = cfg.RunConfig(questions_per_axis=200, tag="main")
    scored, summary = rp.run_all(run)
"""

from . import benchmarks, config, data, evaluate, judge, run_pipeline, steering, vectors

__all__ = [
    "benchmarks",
    "config",
    "data",
    "evaluate",
    "judge",
    "run_pipeline",
    "steering",
    "vectors",
]

__version__ = "1.0.0"
