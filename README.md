## EEG-VJEPA: Self supervised learning from EEG data

### Method
Learning useful representations from unlabeled EEG data via self-supervised pre-training (TUH + NMT). 

Supervised fine-tuning + evaluation done on labeled EEG samples (TUH held-out).

---


### Pre-trained Weights
The pre-trained weights for the following models (best performing models):
- eeg_vjepa_ViT-B_4×30×2
- eeg_vjepa_ViT-M_4×30×4

are available at:

https://huggingface.co/amir-hlp/EEG-VJEPA

---

### Setup

Run:
```bash
conda create -n eeg-vjepa python=3.9 pip
conda activate eeg-vjepa
python setup.py install
```

### Quick start to load the pre-trained weights

```python
import torch
import src.models.vision_transformer as vit
import yaml
import argparse

# Set up argument parser for config file
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='/path/to/configs/pretrain/config.yaml',
                    help='Path to config file')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
model_name = args.get('model').get('model_name',
                                   'vit_small')
cfgs_data = args.get('data')
crop_size = cfgs_data.get('crop_size', (19, 500))
patch_size = cfgs_data.get('patch_size', (4, 30))
if isinstance(patch_size, list):
    patch_size = tuple(patch_size)
num_frames = cfgs_data.get('num_frames', 32)
tubelet_size = cfgs_data.get('tubelet_size', 4)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize encoder
encoder = vit.__dict__[model_name](
    img_size=crop_size,
    patch_size=patch_size, 
    num_frames=num_frames,
    tubelet_size=tubelet_size,
    uniform_power=False,
    use_sdpa=False
).to(device)

# Load pretrained weights
checkpoint = torch.load("path/to/weights.pth.tar",
                        map_location=device)
pretrained_dict = checkpoint['target_encoder']
pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(pretrained_dict, strict=False)

# Set to eval mode for inference
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# Data loader loop
...
```


### Local pre-training

```bash
python -m app.main \
  --fname configs/pretrain/config.yaml \
  --devices cuda:0 cuda:1 cuda:2
```


### Eval

```bash
python -m evals.main \
  --fname configs/eval/config.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

---


### Citation
Our EEG adaption, training runs, and evaluations:
```bibtex
Coming soon
```

V-JEPA Paper (for video):
```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv:2404.08471},
  year={2024}
}
```

