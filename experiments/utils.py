import os
import random
from typing import Any

from omegaconf import DictConfig

from torchvision.transforms.v2 import functional as F
import numpy as np
import torch
from pixeltable.type_system import ColumnType
from torch.utils.data import DataLoader, RandomSampler

from any_gold import AnyRawDataset, AnyVisionSegmentationDataset
from pixeltable import catalog

SEED = int(os.environ.get("SEED", 42))


def force_seed(seed: int = SEED):
    """
    Force the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pxt_run_name(cfg: DictConfig) -> str:
    """
    Generate a unique name for the PixelTable based on the configuration.
    """
    if cfg.pixeltable.run_name is not None:
        return f"{cfg.pixeltable.dir_name}.{cfg.run_name}"

    dataset = cfg.dataset.args._target_.split(".")[-1]
    model = cfg.model.name
    split = cfg.dataset.args.split

    run_name = f"{cfg.pixeltable.dir_name}.{dataset}_{model}_{split}"
    if "category" in cfg.dataset.args:
        run_name += f"_{cfg.dataset.args.category}"

    return run_name


def get_pxt_table_name(
    cfg: DictConfig,
    run_name: str,
) -> str:
    """
    Generate a unique name for the PixelTable view based on the configuration.
    """
    view_name = (
        cfg.pixeltable.table_name
        if cfg.pixeltable.table_name is not None
        else f"seed_{cfg.pipeline.seed}_k_{cfg.pipeline.k_shots}"
    )

    return f"{run_name}.{view_name}"


def get_pxt_table_path(
    cfg: DictConfig,
) -> str:
    """
    Generate the path to the PixelTable based on the configuration.
    """
    run_name = get_pxt_run_name(cfg)
    return get_pxt_table_name(cfg, run_name)


def format_for_pixeltable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or (value.ndim == 1 and len(value) == 1):
            return value.item()
        else:
            return F.to_pil_image(value.cpu())
    else:
        return value


def import_any_dataset_to_pixeltable(
    dataset: AnyVisionSegmentationDataset,
    pxt_table: catalog.Table,
    label: str | None = None,
    num_samples: int | None = None,
    batch_size=16,
    num_workers=0,
) -> None:
    """Load a PyTorch dataset into a PixelTable.

    Args:
        dataset: The PyTorch dataset to import from.
        pxt_table: The PixelTable to import the data into.
        label: The name of the label column in the dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
    """
    raw_dataset = AnyRawDataset(dataset)
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    for batch in dataloader:
        for sample in batch:
            to_insert = {}
            for key, value in sample.items():
                value = format_for_pixeltable(value)
                # if the label is a column name, rename this column `label`
                if key == label:
                    key = "label"

                # add missing column from table definition
                if key not in pxt_table.columns:
                    pxt_table.add_columns(
                        {
                            key: ColumnType.from_python_type(
                                type(value),
                                nullable_default=True,
                                allow_builtin_types=True,
                            )
                        }
                    )
                to_insert[key] = value

            # missing label column mean the label should have the specified value
            if "label" not in to_insert:
                to_insert["label"] = label

            pxt_table.insert(**to_insert)
