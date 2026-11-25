# PatchCore Anomaly Detection

一个开箱即用的**无监督异常检测框架**：只用正常样本建模，不需要任何缺陷标注，即可完成「整图判定 + 缺陷定位」，并自带 Web 界面与 HTTP API。

核心算法来自 [PatchCore](https://arxiv.org/abs/2106.08265)，工程侧提供了完整的训练、推理、评估、服务化链路。

---

## 特性

- **零缺陷样本** —— 只训练正常图，缺什么、坏成什么样都不需要事先见到
- **像素级定位** —— 不止告诉你「有问题」，还给出异常热力图与可疑区域框
- **免训练** —— 直接复用 ImageNet 预训练特征，建模 200 张图约 1 分钟，不需要 GPU 训练
- **一套服务多任务** —— 不同品类各自独立的配置、模型、阈值，互不干扰
- **自带界面** —— 图片 / 视频 / 摄像头三种输入，实时推理并可视化
- **可解释的输出** —— 图像级与像素级双阈值，异常分数可直接对照阈值判断

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9 及以上 |
| PyTorch | 1.12 及以上（CUDA 可选，有 GPU 时快 5–10 倍） |
| 显存 | 推理 ≥ 1 GB；建模 ≥ 2 GB |
| 操作系统 | Windows / Linux / macOS 均可 |

## 安装

```bash
git clone <this-repo>
cd patchcore-anomaly-detection
pip install -r requirements.txt
```

首次运行会自动下载 ImageNet 预训练权重（约 260 MB）。

## 快速开始

### 1. 准备数据

目录结构：

```
data/example/
├── train/
│   └── good/            # 正常样本，必须有
│       ├── 001.png
│       └── ...
└── test/
    ├── good/            # 测试用正常样本，可选
    └── defect/          # 缺陷样本，可选（仅评估用）
```

**关键**：`train/good/` 里只能放正常样本。混入缺陷图会污染特征库，导致漏检。

### 2. 建模

```bash
python -m det.train --task example
```

流程是：提取正常图的 patch 特征 → 核心集采样压缩 → 用留出集标定阈值。产物保存在 `models/example/`。

### 3. 启动服务

```bash
# Windows
start_web.bat

# Linux / macOS
./start_web.sh
```

浏览器打开 http://127.0.0.1:8000。

### 4. 命令行推理

```bash
python -m det.infer --task example --image path/to/test.png
python -m det.infer --task example --dir data/example/test --limit 10
```

### 5. 评估

```bash
python -m det.eval --task example
```

输出图像级 AUROC / AP / F1、像素级 AUROC、AUPRO，以及逐类检出统计，落在 `runs/example/`。

## 配置

每个任务一份配置，放在 `configs/<task>.json`。常用字段：

| 字段 | 含义 | 建议值 |
|---|---|---|
| `task` | 任务名，须与文件名一致 | 自定义 |
| `display_name` | 界面显示名 | 自定义 |
| `train_good_dir` | 正常样本目录 | 相对项目根或绝对路径 |
| `model.backbone` | 特征提取主干 | `wide_resnet50_2` |
| `model.layers` | 取哪些层的特征 | `layer2` + `layer3` |
| `model.input_size` | 输入边长 | 细节多的小缺陷用 640，否则 512 |
| `model.coreset_ratio` | 核心集保留比例 | 0.004 起步，召回不足时调大 |
| `model.smoothing_sigma` | 热力图平滑强度 | 纹理类 1.0，结构类 4.0 |
| `model.score_topk` | 取前 k 个 patch 的均值打分 | 30–100 |
| `threshold.normal_quantile` | 正常分数分位点 | 0.995 |
| `threshold.safety_margin` | 阈值安全系数 | 1.0–1.1，误报多时调大 |

改完配置直接重启服务即可生效，不需要重新建模（除非改了 `model.*`）。

## 项目结构

```
├── app/                    FastAPI 服务
│   ├── main.py             路由与接口
│   ├── runtime.py          任务状态管理
│   └── video.py            视频抽帧推理
├── det/                    算法核心
│   ├── patchcore.py        特征提取、核心集、最近邻
│   ├── engine.py           推理引擎、阈值、区域提取
│   ├── train.py            建模入口
│   ├── infer.py            命令行推理
│   ├── eval.py             评估
│   └── config.py           配置加载
├── web/                    前端（原生 HTML/CSS/JS，无需构建）
├── configs/                任务配置
├── models/                 模型产物（不纳入版本控制）
├── runs/                   推理记录、评估结果、报告
└── tools/
    └── smoke_test.py       接口冒烟测试
```

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/api/tasks` | 任务列表与状态 |
| GET | `/api/tasks/{task}/state` | 任务详情（阈值、历史、统计） |
| POST | `/api/infer/{task}` | 上传图片推理（multipart） |
| POST | `/api/infer/{task}/frame` | 单帧推理（base64，供摄像头用） |
| POST | `/api/tasks/{task}/video` | 提交视频任务 |
| GET | `/api/video/{job_id}` | 查询视频任务进度 |
| POST | `/api/tasks/{task}/threshold` | 修改阈值 |
| GET | `/api/tasks/{task}/export?fmt=csv\|json` | 导出历史 |
| POST | `/api/tasks/{task}/clear` | 清空历史 |

完整字段见 [docs/运行说明.md](docs/运行说明.md)。

## 算法简述

PatchCore 的思路可以概括为「把图切成小块，给每块存指纹，新图逐块找最近邻」：

1. 用 ImageNet 预训练的 CNN 提取中间层特征（本实现取 `layer2` + `layer3` 拼接）
2. 把特征图视为一组 patch 向量，用随机投影降维到 128 维
3. 用 k-center 贪心采样挑出代表性 patch，构成「记忆库」
4. 推理时，每个 patch 在记忆库里找最近的邻居，距离即异常分数
5. 分数图上做高斯平滑与边界衰减，取 top-k 均值作为整图分数

**为什么必须无监督**：工业缺陷的长尾极难覆盖——新工艺带来新缺陷形态，标注永远滞后。只用正常样本建模，等于「学会了正常长什么样，其余都是可疑」。

## 测试

```bash
# 服务启动后
python tools/smoke_test.py
```

不需要已建好的模型，会测健康检查、任务列表、阈值读写、导出等基础接口。

## 常见问题

**Q：建模要多久？**
200 张 512×512 的图，GPU 约 65 秒，CPU 约 6 分钟。

**Q：为什么召回率不够高？**
三处可调：提高 `input_size`、降低 `threshold.safety_margin`、增大 `coreset_ratio`。前两项收益最直接。

**Q：误报太多怎么办？**
提高 `safety_margin` 或 `normal_quantile`。注意留出集样本太少时（< 50 张），阈值会贴近极值，建议补数据。

**Q：能识别缺陷类型吗？**
不能。无监督方法只输出「异常/正常」与异常位置，不区分缺陷类别。要分类需要额外训练分类器。

**Q：支持多品类吗？**
支持。每个品类一份 `configs/<task>.json`，各自独立的模型与阈值。

## 许可

MIT License，见 [LICENSE](LICENSE)。

算法参考：

```bibtex
@inproceedings{roth2022patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Jose and others},
  booktitle={CVPR},
  year={2022}
}
```
