import pytest
from omegaconf import DictConfig


@pytest.fixture
def cfg() -> DictConfig:
    from hydra import compose, initialize

    with initialize(config_path="../config", version_base=None):
        return compose(config_name="config")


def test_load_sam2_image_predictor_from_huggingface(cfg: DictConfig) -> None:
    # local import because Sam2 is using hydra which is conflicting with hydra.main
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from experiments.single_click.model import (
        load_sam2_image_predictor_from_huggingface,
    )

    model_config = cfg.model
    model = load_sam2_image_predictor_from_huggingface(
        model_cfg=model_config, device=cfg.device
    )
    assert isinstance(model, SAM2ImagePredictor)
