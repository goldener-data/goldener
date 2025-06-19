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
    from experiments.single_click.pixeltable_udf import (
        image_size,
        connected_components,
        random_points,
        bounding_boxes,
        predict_with_sam,
    )
    from experiments.single_click.load import sam_cache

    force_seed(seed=cfg.pipeline.seed)

    dataset_args = dict(cfg.dataset.args)

    logger.info(f"Loading the model: {cfg.model.name}")
    model_config = cfg.model
    if model_config.name == "sam":
        from experiments.single_click.load import (
            load_sam2_image_predictor_from_huggingface,
        )  # Sam2 is using hydra which is conflicting with hydra.main

        model = load_sam2_image_predictor_from_huggingface(cfg.model, device=cfg.device)
        sam_cache[cfg.model.hf_id] = model
        logger.info(f"Loaded model: {model}")
        transforms = None
    else:
        raise NotImplementedError(f"Model {model_config.name} is not implemented yet.")

    logger.info(f"Loading the dataset: {cfg.dataset.args._target_}")
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
                "mask": pxt.Array,
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

    pxt_columns = pxt_table.columns()
    if "image_size" not in pxt_columns:
        logger.info("Adding image_size columns to the pixeltable table")
        pxt_table.add_computed_column(image_size=image_size(pxt_table.image))

    if "connected_components" not in pxt_columns:
        logger.info("Adding connected_components columns to the pixeltable table")
        pxt_table.add_computed_column(
            connected_components=connected_components(
                pxt_table.mask, cfg.pipeline.min_area
            )
        )

    if "bounding_boxes" not in pxt_columns:
        logger.info("Adding bounding_boxes columns to the pixeltable table")
        pxt_table.add_computed_column(
            bounding_boxes=bounding_boxes(pxt_table.connected_components)
        )

    if "random_points" not in pxt_columns:
        logger.info("Adding random_points columns to the pixeltable table")
        pxt_table.add_computed_column(
            random_points=random_points(
                masks=pxt_table.connected_components,
                boxes=pxt_table.bounding_boxes,
                num_points=cfg.pipeline.num_points,
            )
        )

    if model_config.name == "sam":
        if "sam_logits" not in pxt_columns:
            logger.info("Applying segmentation model to the pixeltable table")
            pxt_table.add_computed_column(
                sam_logits=predict_with_sam(
                    model_id=model_config.hf_id,
                    image=pxt_table.image,
                    boxes=pxt_table.bounding_boxes,
                    points=pxt_table.random_points,
                )
            )


if __name__ == "__main__":
    run_experiment()
