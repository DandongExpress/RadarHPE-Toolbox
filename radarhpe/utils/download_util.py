"""Checkpoint / dataset download helpers.

Pretrained weights and (optionally) small demo datasets are hosted on the
Hugging Face Hub, mirroring IQA-PyTorch's distribution model. Large radar
datasets (HuPR, XRF55, mmRadPose, MMVR) are NOT redistributed here — see
docs/Dataset_Preparation.md for their original download pages and licenses.
"""
import os
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = Path(os.environ.get('RADARHPE_CACHE_DIR', Path.home() / '.cache' / 'radarhpe'))


def load_file_from_url_or_hub(
    path_or_id: str,
    repo_type: str = 'model',
    filename: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> str:
    """Resolve a checkpoint reference to a local file path.

    Accepts, in order of precedence:
      1. An existing local file path -> returned unchanged.
      2. A Hugging Face Hub repo id (e.g. ``DandongExpress/radarhpe-pulse-1f-hupr``)
         -> downloaded (and cached) via ``huggingface_hub.hf_hub_download``.

    Args:
        path_or_id: local path or HF Hub repo id.
        repo_type: HF Hub repo type, one of 'model' or 'dataset'.
        filename: filename within the HF Hub repo (defaults to
            'pytorch_model.bin' for repo ids without an explicit file).
        cache_dir: override the local cache directory.

    Returns:
        Absolute local path to the resolved file.
    """
    local_path = Path(path_or_id)
    if local_path.is_file():
        return str(local_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download pretrained checkpoints. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = hf_hub_download(
        repo_id=path_or_id,
        filename=filename or 'pytorch_model.bin',
        repo_type=repo_type,
        cache_dir=str(cache_dir),
    )
    return resolved
