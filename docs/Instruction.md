# Contribution & Plugin-Authoring Guide

RadarHPE-Toolbox uses a registry pattern (see `radarhpe/utils/registry.py`)
so that new models, datasets, and metrics can be added without touching any
central dispatch code. This mirrors the pattern used by BasicSR / IQA-PyTorch.

## Adding a new model architecture

1. Create `radarhpe/archs/my_model_arch.py`.
2. Subclass `radarhpe.archs.base_model.BaseRadarHPEModel` and implement
   `forward(self, x) -> torch.Tensor` returning `[B, J, 3]` joints.
3. Decorate the class with `@MODEL_REGISTRY.register(name='my_model')`.
4. Set the class-level metadata (`paper`, `datasets`, `input_type`, `metrics`)
   so it shows up correctly in `docs/ModelCard.md` and the CLI.
5. Add an entry to `radarhpe/default_model_configs.yml` for documentation
   purposes (optional but recommended).

No other file needs to change — `radarhpe/archs/__init__.py` auto-imports
every `*_arch.py` file in the folder, which triggers registration.

```python
import torch
from radarhpe.archs.base_model import BaseRadarHPEModel
from radarhpe.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(name='my_model')
class MyModel(BaseRadarHPEModel):
    paper = 'Your Name et al., Venue Year'
    datasets = ('HuPR',)
    input_type = 'rad_single_frame'
    metrics = ('MPJPE', 'PA-MPJPE')

    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        self.net = torch.nn.Linear(64 * 64 * 16, num_joints * 3)

    def forward(self, x):
        b = x.shape[0]
        return self.net(x.reshape(b, -1)).reshape(b, self.num_joints, 3)
```

## Adding a new dataset

Same pattern in `radarhpe/data/my_dataset.py`: subclass
`BaseRadarPoseDataset`, set `dataset_name`, and decorate with
`@DATASET_REGISTRY.register(name='my_dataset')`. Override `_index_samples`
only if none of the three built-in layouts (npz / paired / per-frame) fit
your data.

## Adding a new metric

Add a function to `radarhpe/metrics/pose_metrics.py` (or a new file under
`radarhpe/metrics/`) decorated with `@METRIC_REGISTRY.register(name='my_metric')`.
Metrics should accept `(pred, gt)` tensors and return a scalar `torch.Tensor`.

## Third-party plugins (no fork required)

External packages can register their own archs/datasets/metrics via Python
entry points, without needing to fork this repository — see the
`[project.entry-points."radarhpe.archs"]` section of `pyproject.toml`.

## Pull request checklist

- [ ] New code follows the registry pattern above (no edits to central
      dispatch code required).
- [ ] Added/updated an entry in `docs/ModelCard.md` or
      `docs/Dataset_Preparation.md` as appropriate.
- [ ] Added a minimal test under `tests/` that does not require downloading
      a full dataset (use small synthetic tensors).
- [ ] `pytest tests/` passes locally.
- [ ] Preserved original license headers when porting code from another
      repository, and updated `NOTICE` if a new upstream license applies.

## Code style

- Format with `black` and `isort` (see `dev` extras in `pyproject.toml`).
- Lint with `flake8`.
- Keep the public API (`radarhpe.create_model`, `create_dataset`,
  `create_metric`, `list_models`, ...) stable; prefer additive changes.
