from __future__ import annotations

import numpy as np


class HFImageEncoder:
    """Frozen Transformers image encoder adapter for SigLIP/SigLIP2/DINOv2."""
    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name

    @property
    def provenance(self):
        return {"model": self.model_name, "weights": "frozen pretrained", "device": str(self.device)}

    def encode(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        from PIL import Image
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                pil = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in images[start:start+batch_size]]
                batch = self.processor(images=pil, return_tensors="pt").to(self.device)
                out = self.model(**batch)
                vec = getattr(out, "pooler_output", None)
                if vec is None:
                    vec = out.last_hidden_state[:, 0]
                vec = self.torch.nn.functional.normalize(vec.float(), dim=-1)
                chunks.append(vec.cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), np.float32)
