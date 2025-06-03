from logging import getLogger

import hydra
from omegaconf import DictConfig

import pixeltable as pxt
import fiftyone as fo

from experiments.utils import get_pxt_run_name, get_pxt_table_path

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def show_in_fiftyone(cfg: DictConfig) -> None:
    pxt_run_name = get_pxt_run_name(
        cfg,
    )
    existing_pxt_tables = pxt.list_tables(pxt_run_name, recursive=False)

    pxt_table_name = get_pxt_table_path(cfg)
    if pxt_table_name not in existing_pxt_tables:
        raise ValueError(
            f"PixelTable {pxt_table_name} does not exist. Please run the experiment first."
        )

    logger.info(f"Loading PixelTable: {pxt_table_name}")
    pxt_table = pxt.get_table(pxt_table_name)

    fo_dataset = pxt.io.export_images_as_fo_dataset(
        tbl=pxt_table,
        images=pxt_table.image,
    )

    # Launch the FiftyOne App to explore
    session = fo.launch_app(fo_dataset)
    session.wait()


if __name__ == "__main__":
    show_in_fiftyone()
