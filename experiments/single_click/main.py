from logging import getLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pixeltable as pxt

from experiments.utils import (
    force_seed,
    get_pxt_table_name,
    import_any_dataset_to_pixeltable,
    get_pxt_run_name,
)


logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    force_seed(seed=cfg.pipeline.seed)

    dataset_args = dict(cfg.dataset.args)

    logger.info(f"Loading the model: {cfg.model.name}")
    model_config = cfg.model
    if model_config.name == "sam":
        from experiments.single_click.models.load import (
            load_sam2_image_predictor_from_huggingface,
        )

        model = load_sam2_image_predictor_from_huggingface(cfg.model, device="cpu")
        logger.info(f"Loaded model: {model}")
        transforms = None
    else:
        raise NotImplementedError(f"Model {model_config.name} is not implemented yet.")

    logger.info(f"Loading the dataset: {cfg.dataset}")
    dataset = instantiate(dataset_args, transforms=transforms)

    logger.info("Creating the pixeltable directories and table:")
    pxt_dir_name = cfg.pixeltable.dir_name
    pxt.create_dir(pxt_dir_name, if_exists=cfg.pixeltable.if_exists)

    pxt_run_name = get_pxt_run_name(
        cfg,
    )
    pxt.create_dir(pxt_run_name, if_exists=cfg.pixeltable.if_exists)

    pxt_table_name = get_pxt_table_name(cfg, pxt_run_name)
    existing_pxt_tables = pxt.list_tables(pxt_run_name, recursive=False)

    need_new_table = (
        pxt_table_name not in existing_pxt_tables
        or cfg.pixeltable.if_exists == "replace_force"
    )
    if need_new_table:
        pxt_table = pxt.create_table(
            pxt_table_name,
            schema={
                "image": pxt.Image,
                "mask": pxt.Image,
                "index": pxt.Int,
                "label": pxt.String,
            },
            if_exists=cfg.pixeltable.if_exists,
        )
    else:
        pxt_table = pxt.get_table(pxt_table_name)

    if cfg.dataset.label is not None:
        label = cfg.dataset.label
    elif cfg.dataset.label_col is not None:
        label = cfg.dataset.label_col
    else:
        label = None

    if need_new_table:
        logger.info("Adding data to the pixeltable table:")
        import_any_dataset_to_pixeltable(
            dataset=dataset,
            pxt_table=pxt_table,
            num_samples=cfg.pipeline.k_shots,
            label=label,
            batch_size=cfg.load.batch_size,
            num_workers=cfg.load.num_workers,
        )


if __name__ == "__main__":
    pxt_table = run_experiment()
