# AstroTransient

**Astronomical Transient Classification via Light Curve Analysis**

---

## English

AstroTransient identifies 14 classes of astronomical transients and variable sources from multi-band photometric light curves. The classifier is trained on PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge) data and achieves approximately 82% accuracy on the test set.

**Supported classes**: Type Ia, Type II, and Type Ibc supernovae; superluminous supernovae; tidal disruption events; kilonovae; active galactic nuclei; eclipsing binaries; RR Lyrae and Mira variables; M-dwarfs; microlensing events; calcium-rich transients; and related subtypes.

---

### Getting Started

#### Prerequisites

- Python 3.10 or later

#### Installation & Launch

```bash
pip install -r requirements.txt
python launcher.py
```

The launcher automatically checks dependencies, prepares the environment, and starts the web interface. On first run, you will be prompted whether to download the PLAsTiCC demonstration dataset (~35 MB). This dataset enables the Demo and Benchmark tabs (browsing example light curves and evaluating accuracy). If you choose to skip the download, these two tabs will be unavailable, but you may still use the Upload CSV tab with the included sample file (`sample_lightcurve.csv`) or your own data.

A sample light curve file (`sample_lightcurve.csv`) is included in this directory for testing purposes.

#### Manual Start

```bash
python app.py
```

#### Command-Line Interface

```bash
python predict.py --demo          # Predict random sample objects
python predict.py --file data.csv # Classify custom light curve data
python predict.py --batch         # Evaluate accuracy on test set
```

#### Input Format

CSV files with the following columns:

| Column     | Description                          |
|------------|--------------------------------------|
| `mjd`      | Observation time (Modified Julian Date) |
| `flux`     | Brightness measurement               |
| `flux_err` | Flux uncertainty                     |
| `passband` | Filter wavelength in Angstrom        |

Optional columns: `redshift`, `hostgal_specz`, `hostgal_photoz`. A minimum of five valid rows is required.

---

### Technical Summary

The classifier extracts 47 statistical and photometric features from each light curve: amplitude, skewness, kurtosis, per-band statistics, color indices, autocorrelation, and host galaxy properties. Classification is performed using gradient-boosted decision trees (XGBoost) with class-weighted training. The model was validated on 6,040 training samples across 14 classes, achieving 81.9% overall accuracy and 76.0% macro-averaged F1 score.

---

### Project Structure

```
├── app.py                  # Web interface (Gradio)
├── launcher.py             # Cross-platform launcher
├── predict.py              # Command-line inference
├── requirements.txt        # Python dependencies
├── lightcurve_template.csv # CSV template for custom data
├── models/                 # Trained classifier
└── src/                    # Core library
```

---

### Authors

Built by Daisy Shang and NullTemp (2026), with assistance from DeepSeek.

---

### License

MIT


---

## Chinese / 中文

# AstroTransient

**基于光变曲线分析的天文瞬变源分类工具**

AstroTransient 从多波段测光光变曲线中识别 14 类天文瞬变源与变源。分类器基于 PLAsTiCC（LSST 测光时域分类挑战赛）数据训练，测试集准确率约 82%。

**支持类别**：Ia 型、II 型、Ibc 型超新星；超亮超新星；潮汐瓦解事件；千新星；活动星系核；食双星；天琴座 RR 型变星与米拉变星；M 矮星；微透镜事件；钙富瞬变源及相关亚型。

---

### 快速开始

#### 环境要求

- Python 3.10 或更高版本

#### 安装与启动

```bash
pip install -r requirements.txt
python launcher.py
```

启动器将自动检查依赖环境、准备运行条件并启动网页界面。首次运行时，程序将询问是否下载 PLAsTiCC 示例数据集（约 300 MB）。下载后可使用"随机演示"与"批量测试"功能（浏览示例光变曲线、评估模型准确率）。如选择跳过，这两个功能将不可用，但你仍可在"上传 CSV"页面使用本目录附带的示例文件（`sample_lightcurve.csv`）或自己的数据。

本目录包含一份光变曲线示例文件（`sample_lightcurve.csv`），可用于测试。

#### 手动启动

```bash
python app.py
```

#### 命令行工具

```bash
python predict.py --demo          # 随机样本预测
python predict.py --file data.csv # 对自定义光变曲线数据分类
python predict.py --batch         # 测试集准确率评估
```

#### 输入格式

CSV 文件需包含以下列：

| 列名       | 说明                    |
|------------|-------------------------|
| `mjd`      | 观测时间（简约儒略日）    |
| `flux`     | 亮度测量值               |
| `flux_err` | 流量不确定度             |
| `passband` | 滤光片波长（埃）         |

可选列：`redshift`、`hostgal_specz`、`hostgal_photoz`。至少需要 5 行有效数据。

---

### 技术概要

分类器从每条光变曲线中提取 47 个统计与测光特征：振幅、偏度、峰度、各波段统计量、色指数、自相关及宿主星系属性。分类采用梯度提升决策树（XGBoost），并使用类别加权训练。模型在 6,040 个训练样本、14 个类别上验证，总体准确率为 81.9%，宏观平均 F1 分数为 76.0%。

---

### 项目结构

```
├── app.py                  # 网页界面 (Gradio)
├── launcher.py             # 跨平台启动器
├── predict.py              # 命令行推理
├── requirements.txt        # Python 依赖
├── lightcurve_template.csv # 自定义数据 CSV 模板
├── models/                 # 训练好的分类器
└── src/                    # 核心代码库
```

---

### 作者

本项目由 Daisy Shang 与 NullTemp 于 2026 年构建，DeepSeek 辅助完成。

---

### 许可证

MIT
