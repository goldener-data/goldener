from pathlib import Path

import pixeltable.config as pxt_config

from goldener.config import init


def test_init_with_home(tmp_path: Path):
    init(tmp_path)

    assert pxt_config.Config.get().home == tmp_path