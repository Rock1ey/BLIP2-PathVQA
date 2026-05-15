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


YES_NO_PROMPT = """Answer the medical question according to the pathology image.
Please answer only "yes" or "no".

Question: {question}
Answer:"""

OTHER_PROMPT = """Answer the medical visual question according to the pathology image.
Provide a short and concise medical answer.

Question: {question}
Answer:"""

GENERIC_PROMPT = """Question: {question}
Answer:"""

SHORT_ANSWER_PROMPT = "Question: {question} Short answer:"


def _load_pathvqa_annotations(ann_paths):
    annotations = []
    for ann_path in ann_paths:
        with open(ann_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, list):
            annotations.extend(
                ann for ann in loaded if ann.get("answer_type") != "number"
            )
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


def _normalize_yes_no_answer(answer):
    answer = answer.strip().lower()
    if answer.startswith("yes"):
        return "yes"
    if answer.startswith("no"):
        return "no"
    return answer


def _format_prompt(question, answer_type, prompt_mode="typed"):
    question = question.strip()

    if prompt_mode == "raw":
        return question
    if prompt_mode == "generic":
        return GENERIC_PROMPT.format(question=question)
    if prompt_mode == "short_answer":
        return SHORT_ANSWER_PROMPT.format(question=question)
    if prompt_mode != "typed":
        raise ValueError(
            "Unsupported PathVQA prompt_mode '{}'. Use one of: typed, raw, generic, short_answer.".format(
                prompt_mode
            )
        )

    if answer_type == "yes/no":
        return YES_NO_PROMPT.format(question=question)
    return OTHER_PROMPT.format(question=question)


class PathVQADataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        prompt_mode="typed",
    ):
        self.vis_root = vis_root
        self.annotation = _load_pathvqa_annotations(ann_paths)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt_mode = prompt_mode
        self._add_instance_ids()

    def _image_path(self, img_id):
        split = img_id.split("_", 1)[0]
        return os.path.join(self.vis_root, split, f"{img_id}.jpg")

    def __getitem__(self, index):
        ann = self.annotation[index]

        image = Image.open(self._image_path(ann["img_id"])).convert("RGB")
        image = self.vis_processor(image)

        raw_question = ann["sent"].strip()
        answer_type = ann.get("answer_type", "other")
        answers = _answers_from_label(ann.get("label", {}))
        if answer_type == "yes/no":
            answers = [_normalize_yes_no_answer(answer) for answer in answers]

        return {
            "image": image,
            "text_input": _format_prompt(
                raw_question, answer_type, prompt_mode=self.prompt_mode
            ),
            "text_output": random.choice(answers),
            "answers": answers,
            "weights": [1.0] * len(answers),
            "answer": answers[0],
            "answer_type": answer_type,
            "raw_question": raw_question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }


class PathVQAEvalDataset(PathVQADataset):
    def __getitem__(self, index):
        ann = self.annotation[index]

        image = Image.open(self._image_path(ann["img_id"])).convert("RGB")
        image = self.vis_processor(image)

        raw_question = ann["sent"].strip()
        answer_type = ann.get("answer_type", "other")
        answers = _answers_from_label(ann.get("label", {}))
        if answer_type == "yes/no":
            answers = [_normalize_yes_no_answer(answer) for answer in answers]

        return {
            "image": image,
            "text_input": _format_prompt(
                raw_question, answer_type, prompt_mode=self.prompt_mode
            ),
            "answer": answers[0],
            "answer_type": answer_type,
            "raw_question": raw_question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
