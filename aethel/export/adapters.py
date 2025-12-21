import json
from typing import Any, Dict


def export_adapters(model: Any, path: str):
    # Matryoshka slice mapping for dense embeddings.
    slices = getattr(model.head, "matryoshka_slices", (768, 512, 256, 128))
    adapter: Dict[str, Any] = {
        "dense_dim": slices[0],
        "slices": {str(s): {"start": 0, "end": s} for s in slices},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(adapter, f, indent=2)
    print(f"Saved adapter metadata to {path}")


__all__ = ["export_adapters"]
