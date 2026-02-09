# Generative Recommenders

Repository hosting code for ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations`` ([ICML'24 paper](https://proceedings.mlr.press/v235/zhai24a.html)) and related code, where we demonstrate that the ubiquitously used classical deep learning recommendation paradigm (DLRMs) can be reformulated as a generative modeling problem (Generative Recommenders or GRs) to overcome known compute scaling bottlenecks, propose efficient algorithms such as HSTU and M-FALCON to accelerate training and inference for large-scale sequential models by 10x-1000x, and demonstrate scaling law for the first-time in deployed, billion-user scale recommendation systems.

## Getting started

We recommend using `requirements.txt`. This has been tested with Ubuntu 22.04, CUDA 12.4, and Python 3.10.

```bash
pip3 install -r requirements.txt
```

Alternatively, you can manually install PyTorch based on official instructions. Then,

```bash
pip3 install gin-config pandas fbgemm_gpu torchrec tensorboard
```

## Experiments

### Public Experiments

To reproduce the public experiments in our paper (traditional sequential recommender setting, Section 4.1.1) on MovieLens and Amazon Reviews in the paper, please follow these steps:

#### Download and preprocess data.

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

A GPU with 24GB or more HBM should work for most datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

Other configurations are included in configs/ml-1m, configs/ml-20m, and configs/amzn-books to make reproducing these experiments easier.

#### Verify results.

By default we write experimental logs to exps/. We can launch tensorboard with something like the following:

```bash
tensorboard --logdir ~/generative-recommenders/exps/ml-1m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/ml-20m-l200/ --port 24001 --bind_all
tensorboard --logdir ~/generative-recommenders/exps/amzn-books-l50/ --port 24001 --bind_all
```

With the provided configuration (.gin) files, you should be able to reproduce the following results (verified as of 04/15/2024):

**MovieLens-1M (ML-1M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------| --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2853           | 0.1603          | 0.5474          | 0.2185          | 0.7528          | 0.2498          |
| BERT4Rec      | 0.2843 (-0.4%)   | 0.1537 (-4.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2811 (-1.5%)   | 0.1648 (+2.8%)  |                 |                 |                 |                 |
| HSTU          | 0.3097 (+8.6%)   | 0.1720 (+7.3%)  | 0.5754 (+5.1%)  | 0.2307 (+5.6%)  | 0.7716 (+2.5%)  | 0.2606 (+4.3%)  |
| HSTU-large    | **0.3294 (+15.5%)**  | **0.1893 (+18.1%)** | **0.5935 (+8.4%)**  | **0.2481 (+13.5%)** | **0.7839 (+4.1%)**  | **0.2771 (+10.9%)** |

**MovieLens-20M (ML-20M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2889           | 0.1621          | 0.5503          | 0.2199          | 0.7661          | 0.2527          |
| BERT4Rec      | 0.2816 (-2.5%)   | 0.1703 (+5.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2813 (-2.6%)   | 0.1730 (+6.7%)  |                 |                 |                 |                 |
| HSTU          | 0.3273 (+13.3%)  | 0.1895 (+16.9%) | 0.5889 (+7.0%)  | 0.2473 (+12.5%) | 0.7952 (+3.8%)  | 0.2787 (+10.3%) |
| HSTU-large    | **0.3556 (+23.1%)**  | **0.2098 (+29.4%)** | **0.6143 (+11.6%)** | **0.2671 (+21.5%)** | **0.8074 (+5.4%)**  | **0.2965 (+17.4%)** |

**Amazon Reviews (Books)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------|---------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.0306           | 0.0164          | 0.0754          | 0.0260          | 0.1431          | 0.0362          |
| HSTU          | 0.0416 (+36.4%)  | 0.0227 (+39.3%) | 0.0957 (+27.1%) | 0.0344 (+32.3%) | 0.1735 (+21.3%) | 0.0461 (+27.7%) |
| HSTU-large    | **0.0478 (+56.7%)**  | **0.0262 (+60.7%)** | **0.1082 (+43.7%)** | **0.0393 (+51.2%)** | **0.1908 (+33.4%)** | **0.0517 (+43.2%)** |

for all three tables above, the ``SASRec`` rows are based on [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) but with the original binary cross entropy loss
replaced with sampled softmax losses proposed in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039). These rows are reproducible with ``configs/*/sasrec-*-final.gin``.
The ``BERT4Rec`` and ``GRU4Rec`` rows are based on results reported by [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) -
note that the comparison slightly favors these two, due to them using full negatives whereas the other rows used 128/512 sampled negatives. The ``HSTU`` and ``HSTU-large`` rows are based on [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152); in particular, HSTU rows utilize identical configurations as SASRec. ``HSTU`` and ``HSTU-large`` results can be reproduced with ``configs/*/hstu-*-final.gin``.

### Synthetic Dataset / MovieLens-3B

We support generating synthetic dataset with fractal expansion introduced in https://arxiv.org/abs/1901.08910. This allows us to expand the current 20 million real-world ratings in ML-20M to 3 billion.

To download the pre-generated synthetic dataset:

```bash
pip3 install gdown
mkdir -p tmp/ && cd tmp/
gdown https://drive.google.com/uc?id=1-jZ6k0el7e7PyFnwqMLfqUTRh_Qdumt-
unzip ml-3b.zip && rm ml-3b.zip
```

To generate the synthetic dataset on your own:

```bash
python3 run_fractal_expansion.py --input-csv-file tmp/ml-20m/ratings.csv --write-dataset True --output-prefix tmp/ml-3b/
```

### Efficiency experiments

``ops/triton`` contains triton kernels needed for efficiency experiments. ``ops/cpp`` contains efficient CUDA kernels. In particular, ``ops/cpp/hstu_attention`` contains the attention implementation based on [FlashAttention V3](https://github.com/Dao-AILab/flash-attention) with state-of-the-art efficiency on H100 GPUs.

## DLRM-v3

We have created a DLRM model using HSTU and have developed benchmarks for both training and inference to faciliate production RecSys use cases.

#### Run model training with 4 GPUs

```bash
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py --dataset debug --mode train
```

#### Run model inference with 4 GPUs

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python -m pip install .

LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug
```

## License
This codebase is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.

## Contributors
The overall project is made possible thanks to the joint work from many technical contributors (listed in alphabetical order):

Adnan Akhundov, Bugra Akyildiz, Shabab Ayub, Alex Bao, Renqin Cai, Jennifer Cao, Xuan Cao, Guoqiang Jerry Chen, Lei Chen, Li Chen, Sean Chen, Xianjie Chen, Huihui Cheng, Weiwei Chu, Ted Cui, Shiyan Deng, Nimit Desai, Fei Ding, Shilin Ding, Francois Fagan, Lu Fang, Leon Gao, Zhaojie Gong, Fangda Gu, Liang Guo, Liz Guo, Jeevan Gyawali, Yuchen Hao, Daisy Shi He, Michael Jiayuan He, Yu He, Samuel Hsia, Jie Hua, Yanzun Huang, Hongyi Jia, Rui Jian, Jian Jin, Rafay Khurram, Rahul Kindi, Changkyu Kim, Yejin Lee, Fu Li, Han Li, Hong Li, Shen Li, Rui Li, Wei Li, Zhijing Li, Lucy Liao, Xueting Liao, Emma Lin, Hao Lin, Chloe Liu, Jingzhou Liu, Xing Liu, Xingyu Liu, Kai Londenberg, Yinghai Lu, Liang Luo, Linjian Ma, Matt Ma, Yun Mao, Bert Maher, Ajit Mathews, Matthew Murphy, Satish Nadathur, Min Ni, Jongsoo Park, Colin Peppler, Jing Qian, Lijing Qin, Jing Shan, Alex Singh, Timothy Shi,  Yu Shi, Dennis van der Staay, Xiao Sun, Colin Taylor, Shin-Yeh Tsai, Rohan Varma, Omkar Vichare, Alyssa Wang, Pengchao Wang, Shengzhi Wang, Wenting Wang, Xiaolong Wang, Yueming Wang, Zhiyong Wang, Wei Wei, Bin Wen, Carole-Jean Wu, Yanhong Wu, Eric Xu, Bi Xue, Hong Yan, Zheng Yan, Chao Yang, Junjie Yang, Wen-Yun Yang, Ze Yang, Zimeng Yang, Yuanjun Yao, Chunxing Yin, Daniel Yin, Yiling You, Jiaqi Zhai, Keke Zhai, Yanli Zhao, Zhuoran Zhao, Hui Zhang, Jingjing Zhang, Lu Zhang, Lujia Zhang, Na Zhang, Rui Zhang, Xiong Zhang, Ying Zhang, Zhiyun Zhang, Charles Zheng, Erheng Zhong, Zhao Zhu, Xin Zhuang.

For the initial paper describing the Generative Recommender problem formulation and the algorithms used, including HSTU and M-FALCON, please refer to ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``([ICML'24 paper](https://dl.acm.org/doi/10.5555/3692070.3694484), [slides](https://icml.cc/media/icml-2024/Slides/32684.pdf)).

## Yelp 两次实验复盘（第一次 vs 第二次）

本节记录本仓库在 Yelp 任务上的两次完整实验过程、代码改动点和结果对比，便于复现与审计。

### 1) 运行环境与数据

- 环境：`conda` 环境 `gr26`
- 关键依赖：`torch`、`fbgemm_gpu`、`gin-config`
- 项目目录：`/home/linjx/code/generative-recommenders`
- Yelp 原始数据目录：`/home/linjx/code/SELFRec-main/dataset/yelp`
- 配置文件：`configs/yelp/hstu-sampled-softmax-n512-final.gin`

### 2) 第一次跑模型（Baseline）细节

#### 2.1 当时的训练与评估逻辑

- 预处理会构建 `train/valid/test` 三个 split 的序列文件，但不做“严格时序过滤”。
- Yelp 的训练阶段评估集默认使用 `test`（即训练期间即看测试集表现）。
- 没有基于 `selection_score` 的 best checkpoint 选择与早停控制。

#### 2.2 典型运行方式（第一次）

```bash
conda activate gr26
python preprocess_yelp_data.py
python main.py --gin_config_file configs/yelp/hstu-sampled-softmax-n512-final.gin --master_port=12345
```

训练完成后常用最后一个 checkpoint（如 `*_ep200`）做评估：

```bash
python evaluate_checkpoint.py \
  --gin_config_file configs/yelp/hstu-sampled-softmax-n512-final.gin \
  --checkpoint_path ckpts/yelp-l50/<YOUR_CKPT_EP200> \
  --device cuda
```

### 3) 第二次代码修改（相对第一次）

#### 3.1 严格时序评估预处理（核心改动）

- 文件：`generative_recommenders/research/data/preprocessor.py`
- 改动：新增严格时序过滤，保证 `target_ts > max(train_history_ts)` 才进入 `valid/test`。
- 同时在预处理日志输出过滤统计（保留数/剔除数）。

本次重跑预处理后的日志示例：

- `train/valid/test = 166620/3540/3601`
- 过滤前 `valid/test = 55479/55436`

#### 3.2 训练阶段改为 `valid` 选模，`test` 仅最终汇报

- 文件：`generative_recommenders/research/data/reco_dataset.py`
- 改动：`get_reco_dataset(...)` 新增 `eval_split`，支持 `valid/test` 切换。

- 文件：`generative_recommenders/research/trainer/train.py`
- 改动：`train_fn(...)` 新增 `eval_split` 参数并传入数据集构建逻辑。

- 文件：`configs/yelp/hstu-sampled-softmax-n512-final.gin`
- 改动：`train_fn.eval_split = "valid"`。

#### 3.3 新增 best checkpoint 与早停机制

- 文件：`generative_recommenders/research/trainer/train.py`
- 改动：
  - 新增 `model_selection_weights`
  - 新增 `save_best_checkpoint`
  - 新增 `early_stop_patience / early_stop_min_epochs / early_stop_min_delta / early_stop_full_eval_only`
  - 日志新增 `selection_score`、`NDCG@20`、`HR@20`

- 当前 Yelp 配置：
  - `train_fn.save_best_checkpoint = True`
  - `train_fn.model_selection_weights = {'ndcg@10': 1.0, 'ndcg@20': 2.0, 'hr@10': 1.0, 'hr@20': 2.0}`
  - `train_fn.early_stop_patience = 6`
  - `train_fn.early_stop_min_epochs = 10`
  - `train_fn.early_stop_min_delta = 1e-4`
  - `train_fn.early_stop_full_eval_only = True`

#### 3.4 评估与日志落盘增强

- 文件：`evaluate_checkpoint.py`
- 改动：
  - 支持 `--eval_split {valid,test}`
  - 支持 `--metrics_out` 与 `--key_metrics_out`
  - 增加 CUDA/fbgemm 运行前检查
  - 输出 `recall@10/20` 与 `ndcg@10/20` 等关键指标

- 文件：`run_yelp_train_eval.sh`（新增）
- 改动：
  - 一键完成预处理（按需）+ 训练 + 最终评估
  - `RUN_PREPROCESS=auto` 时自动检查当前缓存是否 strict-chrono；若旧缓存则自动重做预处理
  - 最终评估固定使用 `test`，并落盘日志与指标

### 4) 第二次跑模型（当前推荐流程）

```bash
cd /home/linjx/code/generative-recommenders
conda activate gr26
RUN_PREPROCESS=auto bash run_yelp_train_eval.sh
```

关键产物会落在 `logs/yelp/`：

- `preprocess_<ts>.log`
- `train_<ts>.log`
- `eval_<ts>.log`
- `metrics_<ts>.json`
- `metrics_<ts>.txt`
- `run_<ts>.summary`

### 5) 两次结果对比

定义：

- `Δ = 第二次 - 第一次`
- `相对Δ = Δ / 第一次`

| 指标 | 第一次 | 第二次 | Δ | 相对Δ |
| --- | ---: | ---: | ---: | ---: |
| HR@10 | 0.034869 | 0.046654 | +0.011785 | +33.8% |
| NDCG@10 | 0.016894 | 0.023001 | +0.006107 | +36.1% |
| HR@20 | 0.063298 | 0.072202 | +0.008904 | +14.1% |
| NDCG@20 | 0.024014 | 0.029455 | +0.005441 | +22.7% |
| HR@50 | 0.129032 | 0.138017 | +0.008985 | +7.0% |
| NDCG@50 | 0.036896 | 0.042344 | +0.005448 | +14.8% |
| HR@100 | 0.212768 | 0.234102 | +0.021334 | +10.0% |
| NDCG@100 | 0.050411 | 0.057906 | +0.007495 | +14.9% |
| HR@200 | 0.339887 | 0.373785 | +0.033898 | +10.0% |
| NDCG@200 | 0.068124 | 0.077312 | +0.009188 | +13.5% |
| HR@500 | 0.581644 | 0.608442 | +0.026798 | +4.6% |
| HR@1000 | 0.766163 | 0.785615 | +0.019452 | +2.5% |
| MRR | 0.018700 | 0.023128 | +0.004428 | +23.7% |
| HR@10_>=4 | 0.036125 | 0.048111 | +0.011986 | +33.2% |
| NDCG@10_>=4 | 0.017439 | 0.023263 | +0.005824 | +33.4% |
| HR@50_>=4 | 0.132573 | 0.142073 | +0.009500 | +7.2% |
| MRR_>=4 | 0.019148 | 0.023080 | +0.003932 | +20.5% |

### 6) 结论

- 第二次改造后，核心目标指标 `HR@10/20` 与 `NDCG@10/20` 均明显提升。
- 同时训练/评估协议更规范：训练看 `valid`，最终只在 `test` 汇报。
- 日志、checkpoint 与关键指标均可稳定落盘，便于后续复现实验与回归分析。
