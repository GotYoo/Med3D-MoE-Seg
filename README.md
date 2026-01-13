# Med3D-MoE-Seg

基于 Mixture of Experts (MoE) 的 3D 医学图像分割系统

## 项目概述

Med3D-MoE-Seg 是一个整合了多个先进医学图像处理技术的 3D 分割框架：

- **Vision Encoder**: CT-CLIP (来自 BTB3D)
- **Language Model**: MoE LLaMA (来自 MedPLIB)
- **Mask Decoder**: SAM-Med3D
- **整体架构**: 受 LISA 启发

## 项目结构

```
Med3D-MoE-Seg/
├── assets/                     # 存放示意图等
├── config/                     # 配置文件
│   ├── config.json            # 模型超参数
│   └── dataset_config.json    # 数据路径配置
├── data/                       # 数据处理模块
│   ├── builder.py
│   └── processors/
│       ├── btb3d_processor.py  # 3D NIfTI 数据加载
│       └── sam_3d_transform.py # 数据增强
├── model/                      # 模型模块
│   ├── medplib_core/          # MedPLIB MoE LLaMA
│   ├── encoders/              # Vision Encoders
│   ├── decoders/              # Mask Decoders
│   └── meta_arch/             # 整体架构
├── scripts/                    # 训练/评估脚本
│   ├── train_ds.sh
│   └── eval.sh
├── train_net.py               # 训练入口
└── requirements.txt           # 依赖
```

## 安装

```bash
# 克隆仓库
cd Med3D-MoE-Seg

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
bash scripts/train_ds.sh
```

### 评估

```bash
bash scripts/eval.sh
```

### 自定义配置

编辑 `config/config.json` 和 `config/dataset_config.json` 以修改模型和数据配置。

## 架构说明

```
[3D CT Image] 
    ↓
[CT-CLIP Encoder] (BTB3D)
    ↓
[Vision Features] → [MM Projector] → [MoE LLaMA] (MedPLIB)
    ↓
[Text + Vision Features]
    ↓
[SAM-Med3D Decoder]
    ↓
[Segmentation Mask]
```

## TODO

- [ ] 实现完整的数据加载流程
- [ ] 集成 BTB3D CT-CLIP 模型
- [ ] 集成 MedPLIB MoE LLaMA
- [ ] 集成 SAM-Med3D 解码器
- [ ] 实现训练循环
- [ ] 实现评估指标 (Dice, IoU, etc.)
- [ ] 添加可视化工具

## 引用

如果使用本项目，请引用相关论文：

```bibtex
@article{btb3d,
  title={BTB3D: ...},
  author={...},
  journal={...},
  year={...}
}

@article{medplib,
  title={MedPLIB: ...},
  author={...},
  journal={...},
  year={...}
}

@article{sam_med3d,
  title={SAM-Med3D: ...},
  author={...},
  journal={...},
  year={...}
}
```

## 许可证

[MIT License](LICENSE)
