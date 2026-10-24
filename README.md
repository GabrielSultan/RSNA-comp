# RSNA Knee Abnormality Detection

Detection of **12 knee abnormalities** from MRI. Kaggle rank: **98th of 6,425**, solo. Private leaderboard score: **0.948**.

Competition: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (**organized by the Radiology Society of North America**)

## TL;DR

Each study is an MRI volume (several planes, several contrasts). Reports are free text, in nine languages, with only 58 expert annotations (also called the "gold" studies). I built the labels, the volume preprocessing, six networks with different architectures, and the submission.

The strategy was an ensemble of models with diverse architectures.

The selected submission combines six models, in float32. Five cover the knee with a 130 mm field of view. A sixth, distilled from a public model, looks at a slightly wider field (145 mm). One network sees every knee in the same orientation: right knees are flipped to look like left knees, without swapping medial and lateral.

| | |
|---|---|
| Rank | 98th / 6,425, solo |
| Private score | 0.948 |
| Targets | 12 findings (ligaments, menisci, osteoarthritis, effusion, fracture, …) |
| Input | 6 series per study (sagittal, coronal, axial × fluid / T1) |

## The problem

For each study, the model outputs a probability on 12 findings: ACL, MCL, medial and lateral meniscus, medial, lateral, and patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture.

Two difficulties dominate. MRI studies do not all have the same series, or the same number of slices. And most of the training signal is in the radiology reports, not in a mask drawn by a radiologist.

## What I built

1. **Labels.** A lexicon and negation rules read the reports (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French). A second pass with an LLM fills in ambiguous cases. Uncertain cells are masked in the loss. 58 expert-annotated studies are held as a check, and are also added back into training with a higher sample weight.
2. **Volumes.** Each series is cropped to the knee (130 mm field of view, 145 mm for the distilled model), then read in 2.5D: one slice and its two neighbors. The six series of a study occupy six fixed slots. A missing series stays an empty volume.
3. **Six networks**, trained separately, then combined finding by finding. Inference weights are a stochastic weight average (SWA) over a window of epochs, not the last checkpoint.

Some of the models also see knees from the OAI cohort, whose clinical MOAKS scores are mapped onto the same 12 targets.

## The six models

Inference in float32.

| Model | Backbone | Input | SWA window | Config |
|---|---|---|---|---|
| ViT-k320 | DINOv3 ViT-B/16 | 320 px, 130 mm field | epochs 1–3 | [`configs/oai3/kneenet_vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3.yaml`](configs/oai3/kneenet_vit_k_llrd08_320_v3b_oofmask_v20h_full_oai3.yaml) |
| v20b DINO | ResNet-50 RadImageNet, DINO-pretrained on the MRIs | 224 px, 130 mm field | epochs 1–3 | [`configs/oai3/kneenet_v20b_crop130_dino_ep11_oof_v3b_full_oai3.yaml`](configs/oai3/kneenet_v20b_crop130_dino_ep11_oof_v3b_full_oai3.yaml) |
| v20h distilled | ResNet-50, one head per target, student of a teacher | 224 px, 145 mm field | epochs 7–9 | [`configs/kneenet_v20h_crop145_dino_ep11_v3b_rsna_distil_main_full.yaml`](configs/kneenet_v20h_crop145_dino_ep11_v3b_rsna_distil_main_full.yaml) |
| CoAt-a | CoAtNet-0 | 224 px, 130 mm field | epochs 1–3 | [`configs/oai3/kneenet_coat_a_llrd08_v3b_oofmask_full_oai3.yaml`](configs/oai3/kneenet_coat_a_llrd08_v3b_oofmask_full_oai3.yaml) |
| CNX-r7 | ConvNeXt-Small, ImageNet-22k | 224 px, 130 mm field | epochs 7–10 | [`configs/kneenet_cnx_r7_in22k_v3b_oofmask_full.yaml`](configs/kneenet_cnx_r7_in22k_v3b_oofmask_full.yaml) |
| v20h lat | Same family as v20h, right knees flipped | 224 px, 130 mm field | epochs 7–12 | [`configs/archive/kneenet_v20h_crop130_oofmask_v20b_lat_full.yaml`](configs/archive/kneenet_v20h_crop130_oofmask_v20b_lat_full.yaml) |

The shared graph, behind different encoders: a 2.5D window, attention over slices, and one head per finding. Architecture code is in [`src/models/`](src/models/).

## Best single model: v20h

v20h was the strongest model on its own. It is a ResNet-50 (RadImageNet). Two pooling steps turn a stack of slices into one score per finding, with only the study label as supervision. Code: [`src/models/kneenet_v20h.py`](src/models/kneenet_v20h.py).

**CAM.** Each slice is encoded as a 14×14 map. A class activation map gives every finding its own weights on that grid, then pools the features on the anatomy that finding cares about.

**MIL.** The study is a bag of slices. Attention over all slices and all six series puts the weight on the few slices that support each finding. Empty series are masked out.

## Repository

| Path | Contents |
|---|---|
| [`src/nlp/`](src/nlp/) | Report reading and lexicon |
| [`src/data/`](src/data/) | Volumes, laterality, labels, OAI cohort |
| [`src/models/`](src/models/) | Architectures (one frozen version per file) |
| [`src/train/`](src/train/) | Training loop, distillation, metrics |
| [`configs/`](configs/) | Recipes for the submitted models |
| [`configs/archive/`](configs/archive/) | Recipes from earlier experiments |
| [`notebooks/`](notebooks/) | Data and error exploration |
| [`checkpoints/`](checkpoints/) | Training curves and logs, without the weights |

Weights (`.pt`), the competition MRIs, and the pretrained backbones are not in the repository. `checkpoints/` only holds run history.

## Train again

```bash
pip install -r requirements.txt
python -X utf8 -m src.train.train --config configs/kneenet_cnx_r7_in22k_v3b_oofmask_full.yaml
```

The cropped volumes, the label CSV, and the backbone weights named in the YAML have to be in place first. Install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/). The large training runs were done on cloud GPUs (RunPod). Scripts are in [`runpod/`](runpod/).

## License

The code is released under the [MIT License](LICENSE).
