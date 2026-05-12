"""
PathVQA datasets.

The common PathVQA release stores annotations as pickle files with items like:
{
    "img_id": "train_0422",
    "sent": "Where are liver stem cells (oval cells) located?",
    "label": {"in the canals of hering": 1},
    "question_id": 100422000,
    ...
}

Images are expected under <vis_root>/<split>/<img_id>.jpg.
"""

import os
import pickle
import random

from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset


def _load_pathvqa_annotations(ann_paths):
    annotations = []
    for ann_path in ann_paths:
        with open(ann_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, list):
            annotations.extend(loaded)
        else:
            raise TypeError(f"PathVQA annotation must be a list, got {type(loaded)}.")
    return annotations


def _answers_from_label(label):
    if isinstance(label, dict) and len(label) > 0:
        # Keep all labels sorted by score for deterministic output.
        return [
            answer
            for answer, _ in sorted(label.items(), key=lambda item: item[1], reverse=True)
        ]
    if isinstance(label, str):
        return [label]
    return [""]


class PathVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = _load_pathvqa_annotations(ann_paths)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self._add_instance_ids()

    def _image_path(self, img_id):
        split = img_id.split("_", 1)[0]
        return os.path.join(self.vis_root, split, f"{img_id}.jpg")

    def __getitem__(self, index):
        ann = self.annotation[index]

        image = Image.open(self._image_path(ann["img_id"])).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["sent"])
        answers = _answers_from_label(ann.get("label", {}))

        return {
            "image": image,
            "text_input": question,
            "text_output": random.choice(answers),
            "answers": answers,
            "weights": [1.0] * len(answers),
            "answer": answers[0],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }


class PathVQAEvalDataset(PathVQADataset):
    def __getitem__(self, index):
        ann = self.annotation[index]

        image = Image.open(self._image_path(ann["img_id"])).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["sent"])
        answers = _answers_from_label(ann.get("label", {}))

        return {
            "image": image,
            "text_input": question,
            "answer": answers[0],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
