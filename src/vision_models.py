import threading
from ultralytics import YOLO
from transformers import Owlv2Processor, Owlv2ForObjectDetection


_MODEL_LOCK = threading.Lock()
_MODELS = None


def get_vision_models():
    """
    Lazy-load heavy vision models once.
    Returns: (yolo_model, owl_processor, owl_model)
    """
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    with _MODEL_LOCK:
        if _MODELS is not None:
            return _MODELS

        print("Memuat model YOLO Vision...")
        yolo_model = YOLO("best.pt")

        print("Memuat model OWL-ViT...")
        owl_processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble", backend="torchvision"
        )
        owl_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )

        _MODELS = (yolo_model, owl_processor, owl_model)
        return _MODELS
