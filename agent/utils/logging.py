from __future__ import annotations

import logging
import sys


_ROOT_CONFIGURED = False


def _configure_root(level: str, *, force_level: bool = False) -> None:

    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        if force_level:
            logging.getLogger().setLevel(level.upper())
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h, "_ce_default", False):
            root.removeHandler(h)
    handler._ce_default = True 
    root.addHandler(handler)
    root.setLevel(level.upper())
    _ROOT_CONFIGURED = True


def get_logger(name: str = "causal_error", level: str | None = None) -> logging.Logger:

    _configure_root(level or "INFO")
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level.upper())
    logger.propagate = True
    return logger


def set_global_level(level: str) -> None:

    _configure_root(level, force_level=True)
    logging.getLogger().setLevel(level.upper())
