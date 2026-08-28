from pathlib import Path

import pixeltable as pxt
from pixeltable.config import Config


def init(home: str | Path | None = None) -> None:
    """Initialize Goldener configuration.

    Args:
        home: Optional storage location for Pixeltable.
    """
    if home is not None:
        Config.init({"pixeltable.home": str(home)}, reinit=True)

    pxt.init()