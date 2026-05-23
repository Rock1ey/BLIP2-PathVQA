"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import re
import string
import torch
from collections import Counter
from tqdm import tqdm

from lavis.common.utils import is_convertible_to_int
import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        sample_id_key = "",
        ques_files=dict(),
        anno_files=dict(),
        valid_splits=['val']
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = ques_files
        self.anno_files = anno_files

        # generalize to non coco data
        self.sample_id_key = sample_id_key

        self.valid_splits = valid_splits

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        prompt = run_cfg.get("prompt", "")

        # generalize to non coco data
        sample_id_key = run_cfg.get("sample_id_key", "instance_id")
        ques_files = run_cfg.get("ques_files", dict())
        anno_files = run_cfg.get("anno_files", dict())
        valid_splits = run_cfg.get("valid_splits", ["val"])


        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            sample_id_key = sample_id_key,
            ques_files=ques_files,
            anno_files=anno_files,
            valid_splits=valid_splits
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in self.valid_splits:
                if split not in dataset:
                    print(f"Split {split} not found in {ds_name}.")
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file
                else:
                    if split not in self.ques_files: # precomputed and passed in task builder
                        self.ques_files[split] = os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt', f'{ds_name}_{split}_questions.json')
                        self.anno_files[split] = os.path.join(registry.get_path("cache_root"), f'{ds_name}_gt', f'{ds_name}_{split}_annotations.json')
                        if dist_utils.get_rank() == 0:
                            os.makedirs(os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt'), exist_ok=True)
                            try:
                                convert_to_coco_gt(dataset, self.ques_files[split], self.anno_files[split], split, self.sample_id_key)
                            except:
                                pass # tasks like vizwiz with no gt answer
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            if ques_id != int and is_convertible_to_int(ques_id):
                ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )
            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics

def convert_to_coco_gt(data, outpath_questions, outpath_annotations, split, sample_id_key):
    if split not in data:
        return
    questions_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'questions':[]}
    annotations_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'annotations':[]}
    print("Generating ground truth annotations...")
    for ann in tqdm(data[split]):
        if ann == None:
            continue
        # if ann[sample_id_key] not in img_ids:
        #     continue
        ques_id = ann["question_id"]
        ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
        if ques_id != int and is_convertible_to_int(ques_id):
            ques_id = int(ques_id)
        questions_data["questions"].append({"question": ann["text_input"], "image_id": ann[sample_id_key], "question_id": ques_id})
        annotations_data["annotations"].append({
            "question_type": "" if "question_type" not in ann else ann["question_type"],
            "multiple_choice_answer": ann["answers"][0] if isinstance(ann["answers"], list) else ann["answers"],
            "answers": [{"answer":ans, "answer_id":i} for i,ans in enumerate(ann["answers"])] if isinstance(ann["answers"], list) else [{"answer":ann["answers"], "answer_id":0}], 
            "image_id": ann[sample_id_key], 
            "question_id": ques_id,
            "answer_type": "" if "answer_type" not in ann else ann["answer_type"],
        })
       
    json.dump(questions_data, open(outpath_questions, 'w'))
    print(f"Saved questions data at {outpath_questions}")
    json.dump(annotations_data, open(outpath_annotations, 'w'))
    print(f"Saved annotation data at {outpath_annotations}")

        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")



@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
    
    def build_datasets(self, cfg):
        datasets = BaseTask.build_datasets(self,cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            # added to ensure that the ground truth format of answers is as expected for non-gqa but similar tasks
            gt_ans = vqa_tool.processPunctuation(gt_ans)
            gt_ans = vqa_tool.processDigitArticle(gt_ans)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics


@registry.register_task("path_vqa")
class PathVQATask(VQATask):
    def build_datasets(self, cfg):
        return BaseTask.build_datasets(self, cfg)

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt="",
        )

        pred_qa_pairs = []
        question_ids = samples["question_id"]
        gt_answers = samples["answer"]
        answer_types = samples["answer_type"]

        for pred_answer, ques_id, gt_answer, answer_type in zip(
            answers, question_ids, gt_answers, answer_types
        ):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append(
                {
                    "question_id": ques_id,
                    "pred_ans": pred_answer,
                    "gt_ans": gt_answer,
                    "answer_type": answer_type,
                }
            )

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        return self._report_metrics(result_file=result_file)

    @staticmethod
    def _normalize_text(text):
        text = str(text).lower()
        text = "".join(ch if ch not in string.punctuation else " " for ch in text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    @classmethod
    def _token_f1(cls, pred, gt):
        pred_tokens = cls._normalize_text(pred).split()
        gt_tokens = cls._normalize_text(gt).split()

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _normalized_em(cls, pred, gt):
        return float(cls._normalize_text(pred) == cls._normalize_text(gt))

    @staticmethod
    def _normalize_yes_no(text):
        normalized = str(text).strip().lower()
        normalized = normalized.strip(string.whitespace + string.punctuation)
        normalized = "".join(
            ch if ch not in string.punctuation else " " for ch in normalized
        )
        tokens = normalized.split()
        if tokens and tokens[0] in {"yes", "no"}:
            return tokens[0]
        return None

    @dist_utils.main_process
    def _report_metrics(self, result_file):
        results = json.load(open(result_file, "r"))

        yes_no_total = 0
        yes_no_correct = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        other_total = 0
        other_f1_sum = 0.0
        other_em_sum = 0.0

        for res in results:
            answer_type = res["answer_type"]
            pred = res["pred_ans"]
            gt = res["gt_ans"]

            if answer_type == "yes/no":
                yes_no_total += 1
                pred_yn = self._normalize_yes_no(pred)
                gt_yn = self._normalize_yes_no(gt)

                if pred_yn is not None and pred_yn == gt_yn:
                    yes_no_correct += 1

                if pred_yn == "yes" and gt_yn == "yes":
                    true_positive += 1
                elif pred_yn == "yes" and gt_yn != "yes":
                    false_positive += 1
                elif pred_yn != "yes" and gt_yn == "yes":
                    false_negative += 1

            elif answer_type == "other":
                other_total += 1
                other_f1_sum += self._token_f1(pred, gt)
                other_em_sum += self._normalized_em(pred, gt)

        yes_no_acc = yes_no_correct / yes_no_total * 100 if yes_no_total else 0.0
        denom = 2 * true_positive + false_positive + false_negative
        yes_no_f1 = 2 * true_positive / denom * 100 if denom else 0.0
        other_token_f1 = other_f1_sum / other_total * 100 if other_total else 0.0
        other_norm_em = other_em_sum / other_total * 100 if other_total else 0.0

        metrics = {
            "agg_metrics": (yes_no_f1 + other_token_f1) / 2,
            "yes_no_acc": yes_no_acc,
            "yes_no_f1": yes_no_f1,
            "other_token_f1": other_token_f1,
            "other_norm_em": other_norm_em,
            "yes_no_count": yes_no_total,
            "other_count": other_total,
            "total_count": yes_no_total + other_total,
        }

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics


@registry.register_task("discrn_qa")
class DisCRNTask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )

        if answers == None: # corrupt videos
            return []
            
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs


    def build_datasets(self, cfg):
        datasets = BaseTask.build_datasets(self, cfg)
        return datasets
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]
            
            # gt_ans = [vqa_tool.processPunctuation(g) for g in gt_ans]
            # gt_ans = [vqa_tool.processDigitArticle(g) for g in gt_ans]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            tokenized_pred = pred.strip().split(" ")
            for ans in gt_ans:
                if ans in tokenized_pred:
                    pred = ans
                    break

            vqa_acc = 1 if pred in gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
