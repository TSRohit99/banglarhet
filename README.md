# BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech

[![Dataset DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23162113-blue)](https://doi.org/10.5281/zenodo.23162113)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-tsrohit99%2Fbanglarhet-yellow)](https://huggingface.co/datasets/tsrohit99/banglarhet)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

Official code, split files, and documentation for **BanglaRhet**, an annotated corpus of **30,289** Bangla political speech segments labeled with two independent single-label tasks:

1. **Rhetorical technique** (9 labels, including `None`)
2. **Persuasion technique** (7 labels, including `None`)

This repository accompanies the paper:

> **Rohit Kumar Sen and Anik Chowdhury**, "BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech," arXiv preprint, 2026. [arXiv:2610.09464](https://arxiv.org/abs/2610.09464)

The work is also associated with the 2026 conference version, accepted for presentation at the *2nd International Conference on Advances in Computing, Communication, Electrical, and Smart Systems (iCACCESS)*, Dhaka, Bangladesh.

## Links

| Resource | Link |
|---|---|
| Dataset (Zenodo, archival, versioned DOI) | https://doi.org/10.5281/zenodo.23162113 |
| Zenodo record | https://zenodo.org/records/23162113 |
| Dataset (Hugging Face) | https://huggingface.co/datasets/tsrohit99/banglarhet |
| Annotation guidelines | [`BanglaRhet_Annotation_Guidelines.pdf`](BanglaRhet_Annotation_Guidelines.pdf) |
| Paper (IEEE Xplore) | To appear |
| Paper (arXiv) | https://arxiv.org/abs/2610.09464 |

---

## Repository Contents

```text
banglarhet/
├── README.md
├── LICENSE                          # code license (MIT)
├── BanglaRhet_Annotation_Guidelines.pdf
├── ml_baselines_mt.py               # TF-IDF + Logistic Regression / linear SVM baselines
└── transformers_mt.py               # transformer fine-tuning and evaluation
```

---

## Getting the Data

The dataset is **not stored in this repository**. Use either of the following.

### Option 1: Hugging Face

```python
from datasets import load_dataset

ds = load_dataset("tsrohit99/banglarhet")
print(ds)
```

### Option 2: Zenodo

Download `BanglaRhet.csv` (UTF-8) from the Zenodo record: https://doi.org/10.5281/zenodo.23162113

```python
import pandas as pd

df = pd.read_csv("BanglaRhet.csv", encoding="utf-8")
print(df.shape)  # (30289, 6)
```

### Schema

```text
speaker, source, date, segment_text, rhetorical_technique, persuasion_technique
```

| Field | Description |
|---|---|
| `speaker` | Political speaker or attributed speaker (`unknown_speaker` if not determinable) |
| `source` | Link to the source article |
| `date` | Date associated with the source material |
| `segment_text` | Bangla political speech segment |
| `rhetorical_technique` | Dominant rhetorical technique |
| `persuasion_technique` | Dominant persuasion technique |

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Full annotated corpus | **30,289** segments |
| Source articles processed | **15,032** |
| Coverage | 2014 to 2025 |
| Average segment length | 48.34 words |
| Inter-annotator agreement (500-segment subset) | Cohen's κ = **0.67** (rhetorical), **0.71** (persuasion) |
| Experimental benchmark used in the paper | **22,565** samples |

### Labels

**Rhetorical:** `Contrast`, `Emotional Language`, `Repetition`, `Exaggeration`, `Anecdote`, `Rhetorical Question`, `Metaphor`, `Parallelism`, `None`

**Persuasion:** `Blame Assignment`, `Call to Action`, `Unity Call`, `Emotional Appeal`, `Moral Appeal`, `Logical Appeal`, `None`

Full definitions and boundary rules (e.g., Emotional Language vs. Emotional Appeal) are in the [annotation guidelines](BanglaRhet_Annotation_Guidelines.pdf).

### Full corpus vs. experimental benchmark

> **30,289 = full annotated corpus** (released as-is)
> **22,565 = experimental benchmark** used for the reported results

The benchmark is derived from the full corpus by (1) excluding `Parallelism` (137 instances, too few for stable macro-F1 under an 80/10/10 split), and (2) applying controlled downsampling to ten overrepresented rhetorical-persuasion co-occurrence pairs (Table II). The released corpus is **not** modified by this preprocessing.

---

## Reproducing the Benchmark

### 1. Setup

```bash
git clone https://github.com/TSRohit99/banglarhet.git
cd banglarhet
pip install torch transformers datasets scikit-learn pandas
```

### 2. Build the experimental benchmark

A single random seed (**42**) is used for data splitting, controlled downsampling, and model training.

Steps applied to the full corpus:

1. Remove `Parallelism` (137 instances).
2. Apply the controlled downsampling rules in Table II below.
3. Split **80% / 10% / 10%** into train / validation / test, stratified on the rhetorical label (seed 42). Persuasion labels are encoded separately over the same splits.

#### Table II: Controlled downsampling rules

| Rhetorical Label | Persuasion Label | Removed |
|---|---|---:|
| Contrast | Blame Assignment | 2000 |
| Emotional Language | Blame Assignment | 1571 |
| Emotional Language | Call to Action | 447 |
| Exaggeration | Blame Assignment | 537 |
| Repetition | Blame Assignment | 375 |
| Repetition | Call to Action | 1100 |
| None | Call to Action | 300 |
| None | Blame Assignment | 685 |
| None | Unity Call | 300 |
| Anecdote | Unity Call | 272 |

These reductions were chosen by iterative manual inspection of the full rhetorical-persuasion co-occurrence table rather than by a fixed formula. The exact resulting data is released for reproducibility. An ablation (Table IV in the paper) shows that training without these reductions yields lower macro-F1 for all four transformer models.

### 3. Train transformer models

Training configuration (identical across models):

| Setting | Value |
|---|---|
| Epochs | 10 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 128 tokens |
| Loss | Class-weighted cross-entropy (weights inversely proportional to class frequency) |
| Checkpoint selection | Best validation macro-F1 |
| Seed | 42 |
| Other settings | Hugging Face `Trainer` defaults |

Models evaluated:

| Model | Hugging Face ID |
|---|---|
| BanglaBERT | `csebuetnlp/banglabert` |
| Bangla-BERT-Base | `sagorsarker/bangla-bert-base` |
| SahajBERT | `neuropark/sahajBERT` |
| XLM-RoBERTa-Base | `xlm-roberta-base` |

```bash
python transformers_mt.py
```

`transformers_mt.py` fine-tunes and evaluates the transformer models on both the rhetorical and persuasion tasks.

### 4. Classical baselines

TF-IDF with Logistic Regression and TF-IDF with linear SVM, with regularization strength, vocabulary size, and n-gram range selected by grid search on the validation set.

```bash
python ml_baselines_mt.py
```

---

## Results (Test Set)

| Model | Rhet. Acc | Rhet. Macro-F1 | Pers. Acc | Pers. Macro-F1 |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 0.4599 | 0.4587 | 0.5321 | 0.5270 |
| TF-IDF + SVM | 0.4692 | 0.4624 | 0.5241 | 0.5114 |
| Bangla-BERT-Base | 0.5237 | 0.5285 | 0.5733 | 0.5720 |
| SahajBERT | 0.5853 | 0.5974 | 0.6185 | 0.6227 |
| **BanglaBERT** | **0.6460** | **0.6540** | **0.6619** | **0.6646** |
| XLM-RoBERTa-Base | 0.6043 | 0.6110 | 0.6349 | 0.6405 |

Under a source-grouped split (no article contributes segments to more than one split), BanglaBERT's macro-F1 is 0.6438 (rhetorical) and 0.6538 (persuasion), a drop of only about one point.

---

## Limitations

- **Subjectivity:** rhetorical and persuasive annotation is inherently subjective; κ of 0.67 / 0.71 indicates moderate-to-substantial agreement.
- **Semantic overlap:** categories such as Emotional Language vs. Exaggeration and Emotional Appeal vs. Moral Appeal overlap naturally.
- **Class imbalance:** remains after downsampling; minority and overlapping classes are harder.
- **Manual downsampling:** the Table II reductions were not produced by a parameterized rule.
- **Manual condensation:** passages over 128 tokens were manually condensed; the proportion was not logged.
- **Scope:** 2014 to 2025, primarily Bangladeshi political news.
- **Single-label formulation:** each segment has one dominant label per task, although multiple techniques may be present.

---

## Ethical Use and Copyright

The dataset analyzes linguistic and rhetorical patterns in publicly accessible political speech. It does **not** determine the truthfulness of claims, rank political individuals or groups, or endorse any political position. It is intended for research and education, **not** for political profiling or voter manipulation.

Speech segments were collected from public news articles (including Prothom Alo and Manab Zamin) and 1,890 source links from the Motamot dataset. The authors do not claim ownership of third-party copyright. Users are responsible for ensuring their use of source-derived text complies with applicable law and terms. Where practical, retain the `source` field and cite the original publication.

---

## Licenses

| Component | License |
|---|---|
| Dataset annotations, labels, documentation, and annotation guidelines (original contributions) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Code in this repository | MIT (see [`LICENSE`](LICENSE)) |
| Third-party source text | Remains under the rights of the original publishers |

CC BY 4.0 applies only to material the authors are authorized to license and does not override third-party rights.

---

## Citation

If you use BanglaRhet, please cite **both** the dataset and the paper.

### Dataset (Zenodo)

```bibtex
@dataset{sen2026banglarhet_data,
  author    = {Sen, Rohit Kumar and Chowdhury, Anik},
  title     = {{BanglaRhet: Bangla Political Speech Rhetoric and Persuasion Dataset}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.23162113},
  url       = {https://doi.org/10.5281/zenodo.23162113}
}
```

### Paper

```bibtex
@misc{sen2026banglarhetbenchmarkingclassicaltransformer,
      title={BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech}, 
      author={Rohit Kumar Sen and Anik Chowdhury},
      year={2026},
      eprint={2610.09464},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.09464}, 
}
```

The arXiv preprint is available at <https://arxiv.org/abs/2610.09464>.

---

## Contact

**Rohit Kumar Sen**, rohit.k.sen.neub@gmail.com


Department of Computer Science and Engineering, North East University Bangladesh, Sylhet, Bangladesh.