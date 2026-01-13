# Med3D-MoE-Seg 与 Submodules 集成指南

## 概述

在之前的代码实现中，我重新实现了许多功能。但实际上，您的 `submodules/` 目录下已经有了 **BTB3D** 和 **MedPLIB** 的完整实现，这些代码应该被直接复用。

## 现有 Submodules 资源

### 1. BTB3D (`submodules/BTB3D/`)

提供以下可复用组件：

#### a. CT-CLIP Vision Tower
- **路径**: `report-generation/llava/model/multimodal_encoder/ct_clip.py`
- **类**: `CLIPVisionTower` (实际使用 CTViT 架构)
- **功能**: 3D 医学图像的视觉编码
- **特点**: 使用 `transformer_maskgit.CTViT` 处理 3D CT 图像

#### b. LLaVA 多模态架构
- **路径**: `report-generation/llava/model/llava_arch.py`
- **类**: `LlavaMetaModel`, `LlavaMetaForCausalLM`
- **功能**: 多模态融合基类，包含 `prepare_inputs_labels_for_multimodal()`

#### c. 数据处理
- **路径**: `report-generation/llava/train/train.py`
- **功能**: 完整的训练流程和数据处理

### 2. MedPLIB (`submodules/MedPLIB/`)

提供以下可复用组件：

#### a. MoE LLaMA 实现
- **路径**: `model/medplib/model/language_model/medplib_moe_llama.py`
- **类**: `MedPLIBMoELlamaModel`, `MedPLIBMoELlamaForCausalLM`
- **功能**: 
  - 完整的 DeepSpeed MoE 集成
  - MoE 层的实现（使用 `deepspeed.moe.layer.MoE`）
  - 支持 sparse 和 dense MoE 模式
- **特点**: 已经实现了 Top-K 路由和梯度累积

#### b. MedPLIB 架构
- **路径**: `model/medplib/model/medplib_arch.py`
- **类**: `LlavaMetaModel`, `LlavaMetaForCausalLM`
- **功能**: 
  - Region-based feature extraction
  - Point sampling (用于细粒度特征)
  - Vision projector

#### c. 训练脚本
- **路径**: `train_ds_medplib.py`
- **功能**: 完整的 DeepSpeed 分布式训练流程
- **特点**: 
  - 支持 LoRA
  - 多种损失函数 (CE, Dice, BCE, IoU, Focal)
  - WandB 集成

#### d. SAM 集成
- **路径**: `model/segment_anything_med2d/`
- **功能**: SAM-Med2D 的完整实现

## 建议的代码重构方案

### 方案 1: 直接导入复用（推荐）

修改现有代码以直接导入 submodules：

```python
# model/encoders/ct_clip_adapter.py
import sys
import os
BTB3D_PATH = os.path.join(os.path.dirname(__file__), '../../submodules/BTB3D/report-generation')
sys.path.insert(0, BTB3D_PATH)

from llava.model.multimodal_encoder.ct_clip import CLIPVisionTower as BTB3D_CTCLIPVisionTower

class CTCLIPVisionTower(BTB3D_CTCLIPVisionTower):
    """直接继承 BTB3D 的实现"""
    pass
```

```python
# model/medplib_core/medplib_moe_llama.py
import sys
import os
MEDPLIB_PATH = os.path.join(os.path.dirname(__file__), '../../submodules/MedPLIB')
sys.path.insert(0, MEDPLIB_PATH)

from model.medplib.model.language_model.medplib_moe_llama import (
    MedPLIBMoELlamaModel as MedPLIB_MoELlamaModel,
    MedPLIBMoELlamaForCausalLM as MedPLIB_MoELlamaForCausalLM
)

# 直接使用或继承扩展
class MedPLibMoELlamaModel(MedPLIB_MoELlamaModel):
    """扩展 MedPLIB 的 MoE 实现以支持 3D 分割"""
    pass
```

```python
# model/medplib_core/llava_arch.py
import sys
import os
MEDPLIB_PATH = os.path.join(os.path.dirname(__file__), '../../submodules/MedPLIB')
sys.path.insert(0, MEDPLIB_PATH)

from model.medplib.model.medplib_arch import (
    LlavaMetaModel as MedPLIB_LlavaMetaModel,
    LlavaMetaForCausalLM as MedPLIB_LlavaMetaForCausalLM
)

# 直接使用 MedPLIB 的实现
LlavaMetaModel = MedPLIB_LlavaMetaModel
LlavaMetaForCausalLM = MedPLIB_LlavaMetaForCausalLM
```

### 方案 2: 符号链接（最简单）

创建符号链接直接使用 submodules 代码：

```bash
# 在项目根目录执行
cd /home/wuhanqing/Med3D-MoE-Seg

# 链接 BTB3D 的 CT-CLIP
ln -sf submodules/BTB3D/report-generation/llava model/encoders/btb3d_llava

# 链接 MedPLIB 的模型
ln -sf ../submodules/MedPLIB/model/medplib model/medplib_core/medplib_original
```

### 方案 3: 作为依赖安装

将 submodules 作为可编辑包安装：

```bash
# 安装 BTB3D
cd submodules/BTB3D/report-generation
pip install -e .

# 安装 MedPLIB
cd ../../MedPLIB
pip install -e .
```

## 需要适配的地方

### 1. SAM-Med3D 集成

MedPLIB 提供的是 SAM-Med2D，您需要：

1. 找到或实现 SAM-Med3D（3D 版本）
2. 参考 MedPLIB 的 SAM 集成方式进行适配

```python
# model/decoders/sam_med3d_adapter.py
# 可以参考 submodules/MedPLIB/model/segment_anything_med2d/ 的结构
# 但需要扩展到 3D
```

### 2. 数据处理

BTB3D 的数据处理可以直接复用：

```python
# data/processors/btb3d_processor.py
import sys
sys.path.insert(0, 'submodules/BTB3D/report-generation')

from llava.train.train import preprocess_multimodal
# 或直接使用 BTB3D 的 Dataset 类
```

### 3. 训练脚本

可以直接修改 MedPLIB 的训练脚本：

```bash
# 复制并修改
cp submodules/MedPLIB/train_ds_medplib.py train_med3d_moe_seg.py
# 然后添加 3D 相关的逻辑
```

## 具体修改步骤

### 步骤 1: 更新导入路径

创建一个 `model/__init__.py` 统一管理导入：

```python
# model/__init__.py
import sys
import os

# 添加 submodules 路径
BTB3D_PATH = os.path.join(os.path.dirname(__file__), '../submodules/BTB3D/report-generation')
MEDPLIB_PATH = os.path.join(os.path.dirname(__file__), '../submodules/MedPLIB')

sys.path.insert(0, BTB3D_PATH)
sys.path.insert(0, MEDPLIB_PATH)

# 从 submodules 导入
from llava.model.multimodal_encoder.ct_clip import CLIPVisionTower as CTCLIPVisionTower
from model.medplib.model.language_model.medplib_moe_llama import (
    MedPLIBMoELlamaModel,
    MedPLIBMoELlamaForCausalLM
)
from model.medplib.model.medplib_arch import LlavaMetaModel, LlavaMetaForCausalLM

# 导出
__all__ = [
    'CTCLIPVisionTower',
    'MedPLIBMoELlamaModel',
    'MedPLIBMoELlamaForCausalLM',
    'LlavaMetaModel',
    'LlavaMetaForCausalLM',
]
```

### 步骤 2: 更新 Med3DLISA

```python
# model/meta_arch/med3d_lisa.py
from model import (
    MedPLIBMoELlamaForCausalLM,
    LlavaMetaModel,
    LlavaMetaForCausalLM,
    CTCLIPVisionTower
)

class Med3DLISAModel(LlavaMetaModel, MedPLIBMoELlamaForCausalLM):
    """直接继承 MedPLIB 和 BTB3D 的实现"""
    
    def __init__(self, config):
        super().__init__(config)
        # 只需要添加 3D 分割相关的组件
        self.seg_decoder = SAMMed3DMaskDecoder(...)
```

### 步骤 3: 更新训练脚本

基于 MedPLIB 的训练脚本修改：

```python
# train_net.py
# 参考 submodules/MedPLIB/train_ds_medplib.py
# 添加 3D 数据处理和 SAM-Med3D 的逻辑
```

## 优势

使用 submodules 的优势：

1. **避免重复实现**: 直接使用经过验证的代码
2. **保持一致性**: 与原始论文实现一致
3. **易于更新**: submodules 更新后自动同步
4. **减少维护**: 不需要维护重复代码

## 注意事项

1. **依赖管理**: 确保安装 BTB3D 和 MedPLIB 的所有依赖
2. **路径问题**: 使用相对路径或环境变量管理 submodules 路径
3. **版本兼容**: 确保 submodules 的版本与您的需求兼容
4. **3D 扩展**: SAM-Med2D 需要扩展为 SAM-Med3D

## 下一步行动

建议按以下顺序进行：

1. ✅ 验证 submodules 的代码可以正常运行
2. ✅ 创建符号链接或更新导入路径
3. ✅ 修改 `model/__init__.py` 统一管理导入
4. ✅ 更新 `med3d_lisa.py` 继承 submodules 的类
5. ✅ 适配 SAM-Med3D (3D 版本)
6. ✅ 测试整个流程

## 参考资料

- BTB3D: `submodules/BTB3D/README.md`
- MedPLIB: `submodules/MedPLIB/README.md`
- MedPLIB Training: `submodules/MedPLIB/train_ds_medplib.py`
