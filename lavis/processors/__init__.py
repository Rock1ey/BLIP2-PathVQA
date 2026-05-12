from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
    BlipImageTrainProcessor,
    BlipQuestionProcessor,
)


__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
    "BlipCaptionProcessor",
    "BlipImageEvalProcessor",
    "BlipImageTrainProcessor",
    "BlipQuestionProcessor",
]


def load_processor(name, cfg=None):
    from lavis.common.registry import registry

    return registry.get_processor_class(name).from_config(cfg)
