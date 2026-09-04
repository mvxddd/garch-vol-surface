"""
volsurface — GARCH volatility forecasting and implied-volatility surface
construction.

Quick start
-----------
    from volsurface import Config, run_pipeline

    cfg = Config()
    cfg.data.ticker = "SPY"          # every config field is mutable
    result = run_pipeline(cfg)
    print(result.headline())
"""
from .config import (AnalyticsConfig, Config, DataConfig, GarchConfig,   # noqa: F401
                     OptionsConfig, SurfaceConfig)
from .i18n import LANGUAGES, get_language, set_language, t              # noqa: F401
from .pipeline import PipelineResult, make_all_figures, run_pipeline     # noqa: F401

__version__ = "1.0.0"
__all__ = ["Config", "DataConfig", "GarchConfig", "OptionsConfig",
           "SurfaceConfig", "AnalyticsConfig", "run_pipeline",
           "PipelineResult", "make_all_figures", "set_language",
           "get_language", "LANGUAGES", "t", "__version__"]
