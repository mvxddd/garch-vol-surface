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
from .config import (
                     AnalyticsConfig,
                     Config,
                     DataConfig,
                     GarchConfig,
                     OptionsConfig,
                     SurfaceConfig,
)
from .i18n import LANGUAGES, get_language, set_language, t
from .pipeline import PipelineResult, make_all_figures, run_pipeline

__version__ = "1.0.0"
__all__ = [
                     "LANGUAGES",
                     "AnalyticsConfig",
                     "Config",
                     "DataConfig",
                     "GarchConfig",
                     "OptionsConfig",
                     "PipelineResult",
                     "SurfaceConfig",
                     "__version__",
                     "get_language",
                     "make_all_figures",
                     "run_pipeline",
                     "set_language",
                     "t",
]
