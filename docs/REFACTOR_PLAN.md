# Med3D-MoE-Seg 重构方案

## 问题分析
之前的方案是：尝试从 submodules 导入 → 失败 → 使用 fallback（简化实现）

但这不是最优解！应该**直接将 BTB3D 和 MedPLIB 的代码整合到项目中**。

## 新的重构策略

### 方案 A：完整复制（推荐）
直接将需要的代码文件从 submodules 复制到项目中，修改导入路径。

**优点**：
- 完全独立，无导入问题
- 可以自由修改和优化
- 符合"直接复用代码"的理念

**需要复制的文件**：
```
submodules/MedPLIB/model/medplib/model/
├── language_model/medplib_moe_llama.py  → model/medplib_core/medplib_moe_llama.py
├── medplib_arch.py                       → model/medplib_core/llava_arch.py  
├── multimodal_encoder/                   → model/medplib_core/multimodal_encoder/
├── multimodal_projector/                 → model/medplib_core/multimodal_projector/
└── (其他依赖)

submodules/BTB3D/report-generation/llava/model/
├── multimodal_encoder/ct_clip.py        → model/encoders/ct_clip_vision_tower.py
└── (其他依赖)
```

### 方案 B：最小化集成
只复制核心类和函数，其他依赖使用我们自己的实现或标准库替代。

**实施步骤**：
1. 复制 MedPLIB 的 MoE LLaMA 核心类（已完成 → _medplib_moe_impl.py）
2. 复制 MedPLIB 的 LLaVA 架构核心类
3. 处理依赖：
   - multimodal_encoder → 使用我们的 ct_clip_adapter
   - multimodal_projector → 简单的 MLP 投影层
   - constants → 定义在本地
   - utils → 复制需要的工具函数

### 当前进度
✅ 已复制：
- `_medplib_moe_impl.py` (MoE LLaMA 实现)
- `_medplib_arch_impl.py` (LLaVA 架构)

⏳ 待处理：
- 修复相对导入
- 复制/实现依赖模块
- 更新原始文件使用新的实现

## 建议
采用**方案 B（最小化集成）**：
1. 保留已复制的核心实现
2. 创建本地的依赖模块（简化版）
3. 测试可用性
4. 逐步完善

这样既能复用核心算法，又不会引入过多依赖。
