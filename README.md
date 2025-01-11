# EEG-VJEPA: Self supervised learning from EEG data

## Method
Learning useful representations from unlabeled EEG data via self-supervised pre-training. 

Fine-tuning + evaluation on labeled EEG samples.


### Local pre-training

```bash
python -m app.main \
  --fname configs/pretrain/cluster_vitl16_EEG.yaml \
  --devices cuda:0 cuda:1 cuda:2
```


## Eval

```bash
python -m evals.main \
  --fname configs/eval/cluster_vitt16_EEG.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

---

### Setup

Run:
```bash
conda create -n jepa python=3.9 pip
conda activate jepa
python setup.py install
```


## Citation
Our EEG adaption, training runs, and evaluations:
```bibtex
Coming soon
```

Original V-JEPA Paper:
```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv:2404.08471},
  year={2024}
}
```

