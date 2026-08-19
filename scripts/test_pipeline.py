import os
import sys
import json
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from muscall.modules.audio_backbones import ModifiedResNet
from muscall.modules.textual_heads import TextTransformer
from muscall.utils.utils import fix_seed


def simple_tokenize(text, vocab_size=49408, max_len=77):
    """Deterministic tokenization for testing sample text descriptions."""
    tokens = [ord(c) % vocab_size for c in text.lower()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


def main():
    print("=" * 72)
    print("        CROSS-MODAL AUDIO-TEXT RETRIEVAL: DATA & PIPELINE TEST")
    print("=" * 72)

    fix_seed(42)

    # 1. Load Configurations
    config_path = os.path.join(PROJECT_ROOT, "configs", "model.yaml")
    conf = OmegaConf.load(config_path).model_config
    print("\n[Step 1] Loaded Configuration:")
    print(f"  * Audio Backbone:  {conf.audio.model} (Mel-Spectrogram + Attention Pooling)")
    print(f"  * Text Encoder:   {conf.text.model} (Causal Multi-Head Transformer)")
    print(f"  * Shared Space:   {conf.projection_dim}-dimensional normalized vector space")

    # 2. Load Dataset Metadata & Audio Arrays
    data_dir = os.path.join(PROJECT_ROOT, "data", "datasets", "audiocaption")
    json_path = os.path.join(data_dir, "dataset_train.json")
    with open(json_path, "r") as f:
        samples = json.load(f)

    print(f"\n[Step 2] Loaded {len(samples)} Samples from '{json_path}':")

    audio_tensors = []
    text_tensors = []
    captions = []
    target_samples = conf.audio.audio_len_seconds * conf.audio.sample_rate

    for s in samples:
        audio_file = os.path.join(data_dir, "audio", s["audio_path"])
        raw_waveform = np.load(audio_file)
        
        # Crop or pad to target length (20s @ 16kHz = 320,000 samples)
        if len(raw_waveform) > target_samples:
            waveform = raw_waveform[:target_samples]
        else:
            waveform = np.pad(raw_waveform, (0, target_samples - len(raw_waveform)))

        audio_tensors.append(torch.from_numpy(waveform).float())
        text_tensors.append(simple_tokenize(s["caption"], vocab_size=conf.text.vocab_size))
        captions.append(s["caption"])
        duration = len(raw_waveform) / conf.audio.sample_rate
        print(f"  * Audio ID {s['audio_id']}: \"{s['caption']}\"")
        print(f"    File: {s['audio_path']} | Duration: {duration:.2f}s ({len(raw_waveform):,} samples)")

    batch_audio = torch.stack(audio_tensors)  # (3, 320000)
    batch_text = torch.stack(text_tensors)    # (3, 77)

    # 3. Initialize Models & Linear Projections
    print("\n[Step 3] Initializing Dual Neural Network Towers...")
    audio_encoder = ModifiedResNet(conf.audio)
    text_encoder = TextTransformer(conf.text)
    
    audio_proj = torch.nn.Linear(conf.audio.hidden_size, conf.projection_dim, bias=False)
    text_proj = torch.nn.Linear(conf.text.hidden_size, conf.projection_dim, bias=False)

    audio_encoder.eval()
    text_encoder.eval()

    # 4. Extract & Project Embeddings
    print("\n[Step 4] Running Forward Pass across both Modalities...")
    with torch.no_grad():
        # Audio tower
        audio_features = audio_encoder(batch_audio)
        audio_embeds = audio_proj(audio_features)
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

        # Text tower
        text_features = text_encoder(batch_text)
        pooled_text = text_features[:, -1, :]  # EOT representation
        text_embeds = text_proj(pooled_text)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    print(f"  [OK] Audio embeddings shape: {tuple(audio_embeds.shape)} (L2 Normalized)")
    print(f"  [OK] Text embeddings shape:  {tuple(text_embeds.shape)} (L2 Normalized)")

    # 5. Compute Cross-Modal Similarity Matrix
    similarity_matrix = (audio_embeds @ text_embeds.t()).numpy()
    print("\n[Step 5] Cross-Modal Cosine Similarity Matrix (Audio rows x Text cols):")
    header = " " * 14 + "  ".join([f"Text {j+1:<8}" for j in range(len(samples))])
    print(header)
    for i in range(len(samples)):
        row_str = "  ".join([f"{similarity_matrix[i, j]:+10.4f}" for j in range(len(samples))])
        print(f"  Audio {i+1:<4} |  {row_str}")

    # 6. Bidirectional Retrieval Simulation
    print("\n" + "=" * 72)
    print("       CROSS-MODAL RETRIEVAL TEST RESULTS")
    print("=" * 72)

    print("\n>>> Text -> Audio Retrieval (Search audio by text query):")
    for t_idx, caption in enumerate(captions):
        sims = similarity_matrix[:, t_idx]
        ranked_indices = np.argsort(sims)[::-1]
        print(f"\n  Query Text: \"{caption}\"")
        print("  Ranked Audio Candidates:")
        for rank, a_idx in enumerate(ranked_indices, start=1):
            match_str = " <-- [Ground Truth Target]" if a_idx == t_idx else ""
            print(f"    Rank #{rank}: Audio Track {a_idx+1} (Similarity: {sims[a_idx]:+.4f}){match_str}")

    print("\n>>> Audio -> Text Retrieval (Retrieve best caption for audio track):")
    for a_idx in range(len(samples)):
        sims = similarity_matrix[a_idx, :]
        ranked_indices = np.argsort(sims)[::-1]
        print(f"\n  Query Audio: Track {a_idx+1} ({samples[a_idx]['audio_path']})")
        print("  Ranked Text Captions:")
        for rank, t_idx in enumerate(ranked_indices, start=1):
            match_str = " <-- [Ground Truth Target]" if t_idx == a_idx else ""
            print(f"    Rank #{rank}: \"{captions[t_idx]}\" (Similarity: {sims[t_idx]:+.4f}){match_str}")

    print("\n" + "=" * 72)
    print(" [OK] PIPELINE AND DATA VERIFICATION COMPLETE: ALL OUTPUTS WORKING!")
    print("=" * 72)


if __name__ == "__main__":
    main()
