import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from huggingface_hub import hf_hub_download
from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_sam2_image_predictor_from_huggingface(
    model_cfg: DictConfig,
    device: str = "cuda",
    mode: str = "eval",
) -> SAM2ImagePredictor:
    hf_id = model_cfg.hf_id

    ckpt_path = hf_hub_download(
        repo_id=hf_id, filename=HF_MODEL_ID_TO_FILENAMES[hf_id][1]
    )
    model = instantiate(model_cfg.config.model, _recursive_=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Missing or unexpected keys in state dict to SAM 2 checkpoint."
        )

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return SAM2ImagePredictor(sam_model=model)


sam_cache: dict[str, SAM2ImagePredictor] = {}
