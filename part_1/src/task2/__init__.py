from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dash import Dash

__all__ = ["create_app", "main"]


def create_app(*args: Any, **kwargs: Any) -> "Dash":
    """Create the dashboard app without importing Dash at package import time."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def main() -> None:
    """Run the dashboard, importing the app module only when needed."""
    from .app import main as _main

    _main()
