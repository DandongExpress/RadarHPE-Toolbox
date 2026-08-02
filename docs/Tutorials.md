# Tutorials

Beginner-to-advanced walkthroughs for RadarHPE-Toolbox. All snippets assume
`pip install -e .` has been run from the repository root (see
[README.md](../README.md#installation)).

## 1. List what's available

```python
import radarhpe

print(radarhpe.list_models())    # ['agile_hpe', 'pppr', 'pulse_1f', 'pulse_kf']
print(radarhpe.list_datasets())  # ['hupr', 'mmradpose', 'mmvr', 'xrf55']
print(radarhpe.list_metrics())   # ['akv', 'mpjpe', 'mpjve', 'pa_mpjpe']
```

## 2. Load a dataset

```python
from torch.utils.data import DataLoader
import radarhpe

train_set = radarhpe.create_dataset('hupr', root='data/HuPR', split='train')
loader = DataLoader(train_set, batch_size=8, shuffle=True)

batch = next(iter(loader))
print(batch['rad'].shape)   # [8, 64, 64, 16]
print(batch['pose'].shape)  # [8, 17, 3]
```

See [Dataset_Preparation.md](Dataset_Preparation.md) for how to obtain and
lay out each supported dataset first.

## 3. Run inference with a pretrained checkpoint

```bash
python -m radarhpe.inference -m pulse_1f -r frame.npy --ckpt checkpoints/pulse_1f_hupr_best.pth
```

or from Python:

```python
import numpy as np
import torch
import radarhpe

model = radarhpe.create_model('pulse_1f', pretrained='checkpoints/pulse_1f_hupr_best.pth', device='cpu')
rad = torch.from_numpy(np.load('frame.npy')).float().unsqueeze(0)
joints = model.predict(rad)   # [1, 17, 3], mm, pelvis-centred
```

## 4. Train a model from a config

```bash
python train.py --config options/train/pulse_1f_hupr.yml
```

Edit the YAML (or copy it) to point at your own dataset root, change the
batch size, or switch `model.name` to `agile_hpe` / `pppr` — the same script
works for every registered model since it only talks to the `radarhpe`
registries, not any model-specific code.

## 5. Evaluate

```bash
python eval.py --config options/test/pulse_1f_hupr.yml --ckpt checkpoints/pulse_1f_hupr_best.pth
```

Reports every metric listed in the config's `metrics:` field (MPJPE,
PA-MPJPE, MPJVE by default for PULSE).

## 6. Add your own model (5-minute plugin)

```python
# radarhpe/archs/my_model_arch.py
import torch
from radarhpe.archs.base_model import BaseRadarHPEModel
from radarhpe.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(name='my_model')
class MyModel(BaseRadarHPEModel):
    paper = 'Your Name et al., Venue Year'
    datasets = ('HuPR',)

    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        self.net = torch.nn.Linear(64 * 64 * 16, num_joints * 3)

    def forward(self, x):
        b = x.shape[0]
        return self.net(x.reshape(b, -1)).reshape(b, self.num_joints, 3)
```

That's it — `radarhpe.create_model('my_model')` now works, no other file
needs to change. See [Instruction.md](Instruction.md) for the full
plugin-authoring guide (datasets, metrics, third-party entry points).

## 7. Compare against the paper baselines

Each model's `metrics` class attribute documents which metrics the original
paper reported it on (see `docs/ModelCard.md`). Compute all of them for a
fair, apples-to-apples comparison:

```python
import radarhpe

mpjpe = radarhpe.create_metric('mpjpe')
pa_mpjpe = radarhpe.create_metric('pa_mpjpe')
mpjve = radarhpe.create_metric('mpjve')

print(mpjpe(pred_pose, gt_pose).item())
print(pa_mpjpe(pred_pose, gt_pose).item())
print(mpjve(pred_seq, gt_seq).item())
```
