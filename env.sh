# 克隆项目
git clone 

# 在项目根目录下执行
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .

# 临时环境变量
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export LAVIS_MODEL_CACHE=/root/autodl-tmp/checkpoints

# 零样本测试 BLIP-2 FLAN-T5-XL
bash run_scripts/blip2/eval/eval_pathvqa_zeroshot_flant5xl.sh

# 微调后测试：
bash run_scripts/blip2/train/train_pathvqa_flant5xl.sh
