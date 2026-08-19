import os
import sys
import json
import argparse
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
    """Tokenize custom text input for the TextTransformer."""
    tokens = [ord(c) % vocab_size for c in text.lower()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


def load_audio_file(file_path, target_samples=320000, sr=16000):
    """Load audio from .npy, .wav, .mp3, etc. and normalize to 20s @ 16kHz."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.endswith(".npy"):
        waveform = np.load(file_path)
    else:
        import librosa
        waveform, _ = librosa.load(file_path, sr=sr)

    # Pad or crop
    if len(waveform) > target_samples:
        waveform = waveform[:target_samples]
    else:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)))

    return torch.from_numpy(waveform).float()


class InteractiveRetrievalEngine:
    def __init__(self):
        fix_seed(42)
        config_path = os.path.join(PROJECT_ROOT, "configs", "model.yaml")
        self.conf = OmegaConf.load(config_path).model_config
        self.target_samples = self.conf.audio.audio_len_seconds * self.conf.audio.sample_rate

        # Initialize Encoders
        self.audio_encoder = ModifiedResNet(self.conf.audio)
        self.text_encoder = TextTransformer(self.conf.text)
        self.audio_proj = torch.nn.Linear(self.conf.audio.hidden_size, self.conf.projection_dim, bias=False)
        self.text_proj = torch.nn.Linear(self.conf.text.hidden_size, self.conf.projection_dim, bias=False)

        self.audio_encoder.eval()
        self.text_encoder.eval()

        # Load known audio database
        self.audio_records = []
        self.audio_embeddings = None
        self._index_dataset()

    def _index_dataset(self):
        data_dir = os.path.join(PROJECT_ROOT, "data", "datasets", "audiocaption")
        json_path = os.path.join(data_dir, "dataset_train.json")
        with open(json_path, "r") as f:
            samples = json.load(f)

        audio_tensors = []
        for s in samples:
            path = os.path.join(data_dir, "audio", s["audio_path"])
            tensor = load_audio_file(path, self.target_samples, self.conf.audio.sample_rate)
            audio_tensors.append(tensor)
            self.audio_records.append({
                "audio_id": s["audio_id"],
                "path": s["audio_path"],
                "default_caption": s["caption"]
            })

        batch_audio = torch.stack(audio_tensors)
        with torch.no_grad():
            feat = self.audio_encoder(batch_audio)
            proj = self.audio_proj(feat)
            self.audio_embeddings = proj / proj.norm(dim=-1, keepdim=True)

    def encode_text(self, text_query):
        """Encode custom user text query into shared embedding."""
        token_tensor = simple_tokenize(text_query, self.conf.text.vocab_size).unsqueeze(0)
        with torch.no_grad():
            feat = self.text_encoder(token_tensor)
            pooled = feat[:, -1, :]
            proj = self.text_proj(pooled)
            embed = proj / proj.norm(dim=-1, keepdim=True)
        return embed

    def search_by_text(self, text_query):
        """Search all indexed audio tracks with a custom text query."""
        text_embed = self.encode_text(text_query)
        # Cosine similarity
        similarities = (self.audio_embeddings @ text_embed.t()).squeeze(1).numpy()
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank": rank,
                "audio_id": self.audio_records[idx]["audio_id"],
                "path": self.audio_records[idx]["path"],
                "default_caption": self.audio_records[idx]["default_caption"],
                "similarity": float(similarities[idx]),
            })
        return results


def run_cli():
    parser = argparse.ArgumentParser(description="Manual & Interactive Cross-Modal Audio-Text Retrieval")
    parser.add_argument("--query", "-q", type=str, default=None, help="Custom text query to search audio database")
    args = parser.parse_args()

    engine = InteractiveRetrievalEngine()

    if args.query:
        # One-shot query mode
        print("\n" + "=" * 65)
        print(f"SEARCHING AUDIO FOR CUSTOM QUERY: \"{args.query}\"")
        print("=" * 65)
        results = engine.search_by_text(args.query)
        for r in results:
            print(f"Rank #{r['rank']}: Audio Track {r['audio_id']} ({r['path']})")
            print(f"  * Similarity Score: {r['similarity']:+.4f}")
            print(f"  * Associated Tag:   \"{r['default_caption']}\"\n")
        return

    # Interactive prompt mode
    print("=" * 70)
    print("       MANUAL INTERACTIVE AUDIO-TEXT RETRIEVAL TOOL")
    print("=" * 70)
    print(f"Indexed {len(engine.audio_records)} audio files ready for search.")
    print("Type any text description to search, or type 'exit' or 'q' to quit.\n")

    while True:
        try:
            user_input = input("Enter your custom text query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            results = engine.search_by_text(user_input)
            print("-" * 65)
            print(f"Results for query: \"{user_input}\"")
            print("-" * 65)
            for r in results:
                print(f"Rank #{r['rank']} | Score: {r['similarity']:+.4f} | Audio ID: {r['audio_id']} ({r['path']})")
                print(f"       Note: \"{r['default_caption']}\"")
            print("-" * 65 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    run_cli()
