# AstroTransient

**Astronomical Transient Classification via Light Curve Analysis**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A 14-class astronomical transient and variable source classifier based on gradient-boosted decision trees (XGBoost). Trained on PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) data, achieving ~82% accuracy on the test set.

---

## Download

Pre-packaged ZIP files ready to use:

- **[deliver.zip](releases/deliver.zip)** (7.2 MB) — Bilingual edition (English / Chinese)
- **[deliver_en.zip](releases/deliver_en.zip)** (34.5 MB) — English-only edition

> See the [included README](releases/README.md) for usage instructions.

---

## Repository Structure

```
├── software/                    # Complete application source code
│   ├── app.py                   # Gradio web GUI (bilingual EN/ZH)
│   ├── launcher.py              # Cross-platform one-click launcher
│   ├── predict.py               # CLI inference tool
│   ├── deliver/                 # [Distribution package (bilingual)](software/deliver)
│   ├── deliver_en/              # [Distribution package (English only)](software/deliver_en)
│   ├── models/                  # All trained model files
│   ├── src/                     # Core library
│   ├── scripts/                 # Training scripts (development history)
│   └── notebooks/               # Jupyter tutorials
│
├── docs/                        # Project documentation
│   ├── [development_log_en.md](docs/development_log_en.md)  # Full development log (EN)
│   ├── [development_log_zh.md](docs/development_log_zh.md)  # Full development log (ZH)
│   ├── [project_report_en.md](docs/project_report_en.md)    # Research report (EN)
│   └── [project_report_zh.md](docs/project_report_zh.md)    # Research report (ZH)
│
├── releases/                    # Downloadable ZIP packages
│   ├── [deliver.zip](releases/deliver.zip)
│   └── [deliver_en.zip](releases/deliver_en.zip)
│
└── README.md                    # This file
```

## Quick Start

```bash
cd software
pip install -r requirements.txt
python launcher.py
```

The launcher handles dependency installation, dataset download (optional ~35 MB), and server startup automatically.

## Supported Classes

Type Ia, II, Ibc supernovae; superluminous supernovae; tidal disruption events; kilonovae; active galactic nuclei; eclipsing binaries; RR Lyrae and Mira variables; M-dwarfs; microlensing events; calcium-rich transients; and related subtypes.

## Technical Summary

- **Features**: 47 statistical and photometric features (global stats, per-band, color indices, host galaxy)
- **Model**: XGBoost (500 trees, max_depth=10)
- **Training**: 6,040 labeled objects, class-weighted loss
- **Performance**: 81.9% accuracy, 76.0% macro-averaged F1 on held-out test set (1,295 objects, 14 classes)

## Authors

**Daisy Shang, NullTemp** — 2026

Developed with assistance from DeepSeek.

## License

MIT


---

## 中文

# AstroTransient

**基于光变曲线分析的天文瞬变源分类工具**

AstroTransient 是一个基于梯度提升决策树（XGBoost）的 14 类天文瞬变源与变源分类器。基于 PLAsTiCC（LSST 测光时域分类挑战赛）数据训练，测试集准确率约 82%。

---

### 下载

预打包的 ZIP 文件，解压即用：

- **[deliver.zip](releases/deliver.zip)**（7.2 MB）— 中英双语版
- **[deliver_en.zip](releases/deliver_en.zip)**（34.5 MB）— 纯英文版

> 使用说明请查看 [附带的 README](releases/README.md)。

---

**支持类别**：Ia 型、II 型、Ibc 型超新星；超亮超新星；潮汐瓦解事件；千新星；活动星系核；食双星；天琴座 RR 型变星与米拉变星；M 矮星；微透镜事件；钙富瞬变源及相关亚型。

---

## 项目结构

```
├── software/                    # 完整应用源代码
│   ├── app.py                   # Gradio 网页界面（中英双语）
│   ├── launcher.py              # 跨平台一键启动器
│   ├── predict.py               # 命令行推理工具
│   ├── deliver/                 # [分发包（双语版）](software/deliver)
│   ├── deliver_en/              # [分发包（纯英文版）](software/deliver_en)
│   ├── models/                  # 全部模型文件
│   ├── src/                     # 核心代码库
│   ├── scripts/                 # 训练脚本（开发记录）
│   └── notebooks/               # Jupyter 教程
│
├── docs/                        # 项目文档
│   ├── [development_log_en.md](docs/development_log_en.md)  # 完整开发日志（英文）
│   ├── [development_log_zh.md](docs/development_log_zh.md)  # 完整开发日志（中文）
│   ├── [project_report_en.md](docs/project_report_en.md)    # 科研报告（英文）
│   └── [project_report_zh.md](docs/project_report_zh.md)    # 科研报告（中文）
│
├── releases/                    # 可下载的 ZIP 包
│   ├── [deliver.zip](releases/deliver.zip)
│   └── [deliver_en.zip](releases/deliver_en.zip)
│
└── README.md                    # 本文件
```

## 快速开始

```bash
cd software
pip install -r requirements.txt
python launcher.py
```

启动器将自动处理依赖安装、数据集下载（可选，约 35 MB）和服务器启动。

---

## 技术概要

- **特征**：47 个统计与测光特征（全局统计、分波段、色指数、宿主星系）
- **模型**：XGBoost（500 棵树，max_depth=10）
- **训练**：6,040 个标注天体，类别加权损失
- **性能**：保留测试集（1,295 个天体，14 类）上准确率 81.9%，宏观平均 F1 76.0%

## 作者

**Daisy Shang, NullTemp** — 2026

本项目在 DeepSeek 的辅助下完成开发。

## 许可证

MIT
