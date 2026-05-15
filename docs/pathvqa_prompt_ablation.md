# PathVQA Prompt Ablation

This branch keeps the `path_vqa` task and metrics unchanged, and changes only
the text fed to the model through `datasets.path_vqa.prompt_mode`.

Supported modes:

- `typed`: answer-type-specific medical prompts.
- `raw`: the original PathVQA question only.
- `generic`: `Question: {question}\nAnswer:`.
- `short_answer`: old LAVIS-style `Question: {question} Short answer:`.

Recommended fine-tuning runs on the 2x RTX 4090 AutoDL server:

```bash
bash run_scripts/blip2/train/train_pathvqa_flant5xl_typed_prompt.sh
bash run_scripts/blip2/train/train_pathvqa_flant5xl_raw_question.sh
bash run_scripts/blip2/train/train_pathvqa_flant5xl_short_answer_prompt.sh
```

Recommended zero-shot prompt ablation:

```bash
bash run_scripts/blip2/eval/eval_pathvqa_zeroshot_flant5xl.sh
bash run_scripts/blip2/eval/eval_pathvqa_zeroshot_flant5xl_raw_question.sh
bash run_scripts/blip2/eval/eval_pathvqa_zeroshot_flant5xl_short_answer_prompt.sh
```

Compare the same metrics across runs:

- `agg_metrics`
- `yes_no_acc`
- `yes_no_f1`
- `other_token_f1`
- `other_norm_em`
