# Contrastive Audio-Text Learning for Cross-Modal Retrieval

Built a dual-encoder framework that aligns audio clips and natural language descriptions in a shared embedding space, enabling bidirectional retrieval between audio and text — no labels, no fine-tuning at inference.

---

## Motivation

Existing audio search systems rely on tags or metadata. This project explores whether a model can learn to match free-form text descriptions to audio purely from (audio, caption) pairs — the same way CLIP works for images, applied to the audio domain.

---

## What I built

```
Audio clip ──► Audio Encoder ──► audio embedding ──┐
                                                    ├──► Symmetric InfoNCE Loss
Text caption ──► Text Encoder ──► text embedding ──┘
```

- **Audio Encoder** — CNN backbone over raw waveforms resampled to 16kHz
- **Text Encoder** — pretrained language model producing sentence-level embeddings
- Both towers project into a shared 512-dim L2-normalised embedding space
- **Loss** — symmetric InfoNCE: matched pairs pulled together, all N-1 negatives in the batch pushed apart
- **Retrieval** — cosine similarity over the embedding space at test time, no extra computation

---

## Results

Evaluated on held-out (audio, caption) pairs in both retrieval directions:

| Direction | R@1 | R@5 | R@10 | Median Rank |
|-----------|-----|-----|------|-------------|
| Text → Audio | 25.9 | 51.9 | 63.3 | 5 |
| Audio → Text | 25.8 | 53.0 | 63.0 | 5 |

Also tested zero-shot transfer to genre classification and auto-tagging — no additional training, class labels used directly as text queries.

---

## Project structure

```
├── configs/                   # All yaml configs
│   ├── datasets/              # Dataset paths and parameters
│   ├── models/                # Encoder architecture configs
│   └── training/              # Learning rate, batch size, epochs
│
├── data/
│   └── datasets/
│       └── audiocaption/      # run this first to test setup
│
├── models/                    # Dual encoder + contrastive loss
├── modules/                   # Audio encoder and text encoder definitions
├── trainers/                  # Training loop
├── utils/                     # R@K and Median Rank evaluation
│
├── scripts/                   # Entry points
│   ├── train.py
│   ├── eval.py
│   └── zeroshot_eval.py
│
├── requirements.txt
└── README.md
```

---

## Dataset format

Each split needs a JSON file (train / val / test):

```json
{
  "audio_id": "track_001",
  "caption": "a calm acoustic guitar melody with light percussion",
  "audio_path": "audio/track_001.npy"
}
```

Preprocess audio to `.npy` arrays before training:

```python
import librosa
import numpy as np

audio, sr = librosa.load("track.wav", sr=16000)
np.save("track.npy", audio)
```

Expected folder structure:

```
data/datasets/your_dataset/
├── audio/
│   ├── track_001.npy
│   └── ...
├── dataset_train.json
├── dataset_val.json
└── dataset_test.json
```

A working toy example is already provided in `data/datasets/audiocaption/` — run this before using any real dataset.

---

## Quickstart

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
pip install -e .
```

Train:

```bash
cd scripts/
python train.py
```

Evaluate retrieval:

```bash
python eval.py
```

Zero-shot classification:

```bash
python zeroshot_eval.py
```

---

## Datasets used

- **Song Describer Dataset** — ~1.1k human-written captions for 706 permissively licensed music recordings. Used for evaluation.
  → github.com/ilaria-manco/song-describer-dataset

- **LP-MusicCaps** — LLM-generated captions from MagnaTagATune tags. Used for training at scale.

- **MuMu** — music paired with Amazon review text (~147K tracks). Used for pretraining experiments.

---

## Key design decisions

**Why independent encoders over cross-attention?**
Cross-attention requires both modalities at query time. Independent encoders let you pre-compute and index all audio embeddings offline — retrieval then becomes a single dot product, which scales to large databases.

**Why InfoNCE over triplet loss?**
Triplet loss uses one negative per anchor. InfoNCE treats all other items in the batch as negatives simultaneously, giving a much denser training signal per step.

**Why L2 normalisation?**
Keeps dot product equivalent to cosine similarity, stabilises training, and makes the loss scale-invariant to embedding magnitude.

---

## Requirements

- Python 3.8+
- PyTorch 1.11+
- librosa
- transformers
- numpy, tqdm, pyyaml
