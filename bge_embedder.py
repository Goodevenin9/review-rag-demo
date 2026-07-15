from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LocalBGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", max_length: int = 512, local_files_only: bool = True):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded)
                vectors = output.last_hidden_state[:, 0]
                vectors = F.normalize(vectors, p=2, dim=1)
                embeddings.extend(vectors.cpu().tolist())
        return embeddings
