# 模型文件说明

## 最终模型 (推荐使用)

| 文件 | 说明 | 准确率 |
|------|------|--------|
| `xgboost_final.pkl` | 最终 XGBoost 模型，47 特征，14 类 | 81.9% |
| `label_encoder.pkl` | 标签编码器 (PLAsTiCC ID <-> 0..13) | - |
| `feature_order.pkl` | 特征名顺序列表 | - |

## 实验过程模型

### 阶段 1: 基线探索
| 文件 | 说明 | 准确率 |
|------|------|--------|
| `xgboost_baseline.pkl` | 第一版 XGBoost (20 特征, 14 类) | 54.3% |

### 阶段 2: 特征工程 + 手工调参
| 文件 | 说明 | 准确率 |
|------|------|--------|
| `xgboost_optimized.pkl` | 手工调参版 (46 特征, 合并 train+val) | 82.8% |
| `xgboost_optuna.pkl` | Optuna 超参搜索版 (80 次, CPU) | 82.2% |

### 阶段 3: GPU 深度学习实验 (均未超过 XGBoost)
| 文件 | 说明 | 准确率 |
|------|------|--------|
| `mlp_classifier.pt` | MLP on 特征 (CPU, 100 epochs) | 69.5% |
| `lstm_classifier.pt` | LSTM on 原始序列 (CPU, 31 epochs) | 22.3% |
| `lstm_gpu.pt` | LSTM v2 (GPU, 87 epochs) | 27.0% |
| `mlp_gpu.pt` | MLP v2 (GPU, 82 epochs) | 69.2% |
| `transformer.pt` | Transformer (GPU, 86 epochs) | 26.0% |
| `transformer_gpu.pt` | Transformer v2 (GPU) | 26.0% |
| `heatmap_cnn.pt` | 2D Heatmap CNN (GPU, 102 epochs) | 33.5% |
| `heatmap_cnn_gpu.pt` | Heatmap CNN v2 (GPU) | 33.5% |
| `contrastive_encoder.pt` | SimCLR 对比学习编码器 (GPU, 150 epochs) | 34.4% |

## 使用方法

```python
import joblib
model = joblib.load("models/xgboost_final.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
```
