from __future__ import annotations

import os
import threading
from typing import Any, Optional, Tuple
from ultralytics import YOLO
from transformers import Owlv2Processor, Owlv2ForObjectDetection


_MODEL_LOCK: threading.Lock = threading.Lock()
_MODELS: Optional[Tuple[YOLO, Owlv2Processor, Owlv2ForObjectDetection]] = None

YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "best.pt")
OWL_MODEL_NAME: str = os.getenv("OWL_MODEL_NAME", "google/owlv2-base-patch16-ensemble")


def get_vision_models() -> Tuple[YOLO, Owlv2Processor, Owlv2ForObjectDetection]:
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    with _MODEL_LOCK:
        if _MODELS is not None:
            return _MODELS

        print("Memuat model YOLO Vision...")
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Gagal memuat YOLO model dari '{YOLO_MODEL_PATH}'. "
                f"Pastikan file model ada atau set env YOLO_MODEL_PATH. Error: {e}"
            ) from e

        print("Memuat model OWL-ViT...")
        try:
            owl_processor = Owlv2Processor.from_pretrained(
                OWL_MODEL_NAME, backend="torchvision"
            )
            owl_model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Gagal memuat OWL-ViT model '{OWL_MODEL_NAME}'. "
                f"Periksa koneksi internet atau set env OWL_MODEL_NAME. Error: {e}"
            ) from e

        _MODELS = (yolo_model, owl_processor, owl_model)
        return _MODELS
