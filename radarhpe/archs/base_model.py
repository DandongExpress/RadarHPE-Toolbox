"""Common base class for every registered HPE model in the toolbox.

Every model, regardless of which paper it comes from, exposes the same
minimal contract so that training, evaluation, and inference code can treat
PULSE / Agile-HPE / PPPR-based models interchangeably:

* ``forward(rad)`` — maps a RAD (or PPPR / heatmap / point-cloud) tensor to
  ``[B, J, 3]`` joint coordinates in millimetres, pelvis-centred.
* ``paper`` / ``datasets`` / ``input_type`` / ``metrics`` — class-level
  metadata used by the CLI, docs generator, and ``radarhpe.list_models()``.
* ``load_pretrained(path_or_id)`` — loads a local checkpoint or downloads one
  from the Hugging Face Hub.
"""
from typing import ClassVar, Sequence, Tuple

import torch
import torch.nn as nn

from radarhpe.utils.download_util import load_file_from_url_or_hub


class BaseRadarHPEModel(nn.Module):
    #: Human-readable citation for the originating paper.
    paper: ClassVar[str] = ''
    #: Datasets this model has been evaluated on in the original paper.
    datasets: ClassVar[Tuple[str, ...]] = ()
    #: Expected input tensor semantics, e.g. 'rad_single_frame', 'rad_multi_frame',
    #: 'pppr', 'heatmap', 'point_cloud'.
    input_type: ClassVar[str] = 'rad_single_frame'
    #: Metrics reported in the original paper (see radarhpe.metrics).
    metrics: ClassVar[Sequence[str]] = ('MPJPE', 'PA-MPJPE')
    #: Number of output joints (17 for the COCO-style skeleton used across
    #: all three source papers; override if a subclass differs).
    num_joints: ClassVar[int] = 17

    def forward(self, rad: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement forward(). '
            'See the TODO(integration) markers in the corresponding *_arch.py file.'
        )

    @torch.no_grad()
    def predict(self, rad: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper around ``forward`` that also sets eval mode."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(rad)
        finally:
            self.train(was_training)

    def load_pretrained(self, path_or_id: str, strict: bool = True) -> None:
        """Load weights from a local checkpoint path or a Hugging Face Hub repo id."""
        ckpt_path = load_file_from_url_or_hub(path_or_id, repo_type='model')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def model_info(cls) -> dict:
        """Return the class-level metadata used by docs/ModelCard.md generation."""
        return {
            'name': cls.__name__,
            'paper': cls.paper,
            'datasets': cls.datasets,
            'input_type': cls.input_type,
            'metrics': list(cls.metrics),
            'num_joints': cls.num_joints,
        }
