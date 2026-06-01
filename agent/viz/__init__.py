from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from ..schemas import CausalErrorGraph
from .dot import render_dot
from .html_template import render_html


def render_svg(dot_source: str, *, timeout_s: float = 15.0) -> Optional[str]:
    binary = shutil.which("dot")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-Tsvg"],
            input=dot_source.encode("utf-8"),
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


__all__ = ["render_dot", "render_html", "render_svg"]
