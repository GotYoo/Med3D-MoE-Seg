# Med3D-LISA API 快速参考

## 🎯 核心 API

### 1. 模型初始化

```python
from model.meta_arch.med3d_lisa import Med3DLISA_Full, Med3DLISAConfig

# 创建配置
config = Med3DLISAConfig(
    # LLM 基础配置
    hidden_size=4096,
    num_experts=8,
    num_experts_per_tok=2,
    
    # Stage 1: Alignment
    biobert_model='dmis-lab/biobert-v1.1',
    biobert_freeze_layers=8,
    latent_dim=512,
    alignment_temperature=0.07,
    
    # Stage 2: RAG
    rag_top_k=3,
    rag_num_entries=1000,
    rag_injection_position='prepend',
    
    # Stage 4: Self-Correction
    consistency_threshold=0.7,
    max_correction_iterations=3,
    
    # Loss weights
    lambda_alignment=0.1,
    lambda_dice=1.0,
    lambda_matching=0.5
)

# 初始化模型
model = Med3DLISA_Full(config)
```

---

### 2. 训练 (Training)

```python
# 准备数据
batch = {
    'input_ids': torch.LongTensor,      # [B, L] - 指令文本
    'attention_mask': torch.Tensor,     # [B, L]
    'images': torch.FloatTensor,        # [B, 1, D, H, W] - CT volume
    'clinical_reports': {
        'input_ids': torch.LongTensor,  # [B, L_text] - 临床报告
        'attention_mask': torch.Tensor  # [B, L_text]
    },
    'labels': torch.LongTensor,         # [B, L] - 目标报告
    'masks_gt': torch.FloatTensor       # [B, 1, D, H, W] - Ground truth mask
}

# 前向传播
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    images=batch['images'],
    clinical_reports=batch['clinical_reports'],
    labels=batch['labels'],
    masks_gt=batch['masks_gt'],
    output_hidden_states=True,
    return_dict=True
)

# 获取损失和输出
total_loss = outputs['loss']              # 总损失
lm_loss = outputs['lm_loss']              # 语言模型损失
seg_loss = outputs['seg_loss']            # 分割损失
alignment_loss = outputs['alignment_loss'] # 对齐损失
matching_loss = outputs['matching_loss']   # 匹配损失
pred_masks = outputs['pred_masks']        # 预测掩码 [B, 1, D, H, W]

# 反向传播
total_loss.backward()
optimizer.step()
```

**损失计算公式**:
```
total_loss = lm_loss + λ_dice * seg_loss + λ_align * alignment_loss + λ_match * matching_loss
```

---

### 3. 推理 (Inference)

#### 基础推理（无自我修正）
```python
generated_ids, pred_masks, info = model.generate_with_mask(
    input_ids=prompt_ids,           # [1, L] - 用户指令
    images=ct_volume,               # [1, 1, D, H, W]
    clinical_reports=reports,       # Dict (可选)
    max_new_tokens=512,
    enable_self_correction=False    # 关闭自我修正
)

# 解码文本
report = tokenizer.decode(generated_ids[0])
```

#### 带自我修正的推理（推荐）
```python
generated_ids, pred_masks, correction_info = model.generate_with_mask(
    input_ids=prompt_ids,
    images=ct_volume,
    clinical_reports=reports,
    max_new_tokens=512,
    enable_self_correction=True,    # 启用自我修正
    temperature=0.7,
    top_p=0.9
)

# 检查修正效果
print(f"迭代次数: {correction_info['num_iterations']}")
print(f"最终分数: {correction_info['final_score']:.3f}")
print(f"是否改进: {correction_info['improved']}")

# 查看历史
for i, result in enumerate(correction_info['history']):
    print(f"Iteration {i}: score = {result['consistency_score']:.3f}")
```

---

### 4. RAG 知识库管理

#### 加载预构建知识库
```python
# 访问 RAG retriever
rag_retriever = model.model.rag_retriever

# 加载知识库
rag_retriever.load_knowledge_base('path/to/knowledge_base.pt')
```

#### 构建新知识库
```python
import torch
from model.encoders.biobert_encoder import BioBERTEncoder

# 1. 准备医学知识文本
medical_texts = [
    "Pulmonary nodules are small masses in the lung...",
    "Ground glass opacity indicates...",
    # ... 更多医学知识
]

# 2. 编码知识
biobert = BioBERTEncoder()
knowledge_embeddings = []

for text in medical_texts:
    tokenized = biobert.tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True
    )
    embed = biobert(tokenized['input_ids'], tokenized['attention_mask'])
    knowledge_embeddings.append(embed)

# 3. 保存知识库
knowledge_base = torch.stack(knowledge_embeddings)
rag_retriever.knowledge_base.data = knowledge_base
rag_retriever.save_knowledge_base('path/to/knowledge_base.pt')
```

---

### 5. 组件独立使用

#### BioBERT Encoder
```python
from model.encoders.biobert_encoder import BioBERTEncoder

encoder = BioBERTEncoder(
    model_name='dmis-lab/biobert-v1.1',
    freeze_layers=8
)

# 编码文本
text_embeds = encoder(
    input_ids=tokenized['input_ids'],
    attention_mask=tokenized['attention_mask']
)  # [B, 768]
```

#### Unified Alignment
```python
from model.encoders.uni_alignment import UnifiedAlignmentModule

alignment = UnifiedAlignmentModule(
    image_dim=512,
    text_dim=768,
    latent_dim=512
)

# 对齐图像和文本
outputs = alignment(
    image_features,  # [B, 512]
    text_features,   # [B, 768]
    return_loss=True
)

aligned_image = outputs['image_embeds']    # [B, 512]
aligned_text = outputs['text_embeds']      # [B, 512]
loss = outputs['contrastive_loss']         # Scalar
```

#### Consistency Checker
```python
from model.correction.consistency import ConsistencyChecker

checker = ConsistencyChecker(
    mask_channels=256,
    text_hidden_size=4096,
    embed_dim=512
)

# 检查一致性
outputs = checker(
    mask_output=predicted_mask,    # [B, 1, D, H, W]
    text_embeds=hidden_states,     # [B, L, 4096]
    return_attention=True
)

score = outputs['consistency_score']      # [B, 1] ∈ [0, 1]
attention = outputs['attention_weights']  # [B, 1, L]
```

---

## 📋 数据格式规范

### 输入格式

#### 1. CT Volume (images)
```python
Shape: [B, 1, D, H, W]
dtype: torch.float32
Range: [0, 1] (normalized)
Example: [2, 1, 96, 256, 256]
```

#### 2. Clinical Report (clinical_reports)
```python
{
    'input_ids': torch.LongTensor,      # [B, L_text]
    'attention_mask': torch.Tensor      # [B, L_text]
}
# 使用 BioBERT tokenizer
```

#### 3. User Instruction (input_ids)
```python
Shape: [B, L]
dtype: torch.int64
Example prompt: "Please segment the lung nodule and generate a report. <SEG>"
# 注意: 必须包含 <SEG> token 才会生成分割掩码
```

#### 4. Ground Truth Mask (masks_gt)
```python
Shape: [B, 1, D, H, W]
dtype: torch.float32
Range: {0, 1} (binary mask)
```

### 输出格式

#### 1. Generated Report (generated_ids)
```python
Shape: [B, L_gen]
dtype: torch.int64
# 使用 tokenizer.decode() 解码为文本
```

#### 2. Predicted Mask (pred_masks)
```python
Shape: [B, 1, D, H, W]
dtype: torch.float32
Range: Logits (training) or [0, 1] probabilities (inference)
# 使用 torch.sigmoid() 转换为概率
# 使用 (probs > 0.5) 转换为二值掩码
```

#### 3. Correction Info (correction_info)
```python
{
    'num_iterations': int,           # 实际迭代次数
    'final_score': float,            # 最终一致性分数 [0, 1]
    'history': List[Dict],           # 每次迭代的详细结果
    'improved': bool                 # 是否比初始版本改进
}
```

---

## ⚙️ 配置参数详解

### Med3DLISAConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **LLM 基础配置** ||||
| `hidden_size` | int | 4096 | LLM 隐藏层维度 |
| `num_experts` | int | 8 | MoE 专家数量 |
| `num_experts_per_tok` | int | 2 | 每个 token 激活的专家数 |
| **Stage 1 配置** ||||
| `biobert_model` | str | 'dmis-lab/biobert-v1.1' | BioBERT 模型路径 |
| `biobert_freeze_layers` | int | 8 | 冻结的 BERT 层数 |
| `latent_dim` | int | 512 | 对齐后的潜在空间维度 |
| `alignment_temperature` | float | 0.07 | 对比学习温度参数 |
| **Stage 2 配置** ||||
| `rag_top_k` | int | 3 | 检索的 top-k 知识数 |
| `rag_knowledge_dim` | int | 512 | 知识库嵌入维度 |
| `rag_num_entries` | int | 1000 | 知识库条目数 |
| `rag_injection_position` | str | 'prepend' | 上下文注入位置 |
| **Stage 4 配置** ||||
| `consistency_threshold` | float | 0.7 | 一致性阈值（触发修正） |
| `max_correction_iterations` | int | 3 | 最大修正迭代次数 |
| `consistency_embed_dim` | int | 512 | 一致性检查嵌入维度 |
| **损失权重** ||||
| `lambda_alignment` | float | 0.1 | 对齐损失权重 |
| `lambda_dice` | float | 1.0 | 分割损失权重 |
| `lambda_matching` | float | 0.5 | 匹配损失权重 |

---

## 🎨 使用场景示例

### 场景 1: 肺结节分割 + 报告生成
```python
# 1. 准备提示
prompt = "Please analyze this CT scan, segment any lung nodules, and generate a diagnostic report. <SEG>"
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

# 2. 准备临床历史（可选）
clinical_history = "Patient is a 65-year-old male with a history of smoking."
report_ids = biobert_tokenizer(clinical_history, return_tensors='pt')

# 3. 推理
generated_ids, pred_mask, info = model.generate_with_mask(
    input_ids=input_ids,
    images=ct_volume,
    clinical_reports=report_ids,
    enable_self_correction=True
)

# 4. 后处理
report = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
binary_mask = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()

print(f"Report: {report}")
print(f"Segmentation shape: {binary_mask.shape}")
print(f"Refinement iterations: {info['num_iterations']}")
```

### 场景 2: 多器官分割
```python
prompt = "Segment the liver, kidneys, and spleen in this abdominal CT. <SEG>"

# 注意: 需要模型支持多类别分割
# 当前版本支持单类别，多类别需要扩展 SAM decoder
```

### 场景 3: 质量控制（无分割，仅报告）
```python
prompt = "Generate a quality assessment report for this CT scan."
# 不包含 <SEG> token，模型只生成文本

generated_ids, _, _ = model.generate_with_mask(
    input_ids=input_ids,
    images=ct_volume,
    enable_self_correction=False  # 无掩码时关闭修正
)
```

---

## 🔧 调试和监控

### 启用详细输出
```python
# 1. 查看所有损失项
outputs = model(...)
for key, value in outputs.items():
    if 'loss' in key and value is not None:
        print(f"{key}: {value.item():.4f}")

# 2. 检查路由负载
if 'router_logits' in outputs:
    router_logits = outputs['router_logits']
    expert_usage = torch.argmax(router_logits, dim=-1)
    print(f"Expert usage distribution: {expert_usage.unique(return_counts=True)}")

# 3. 监控自我修正
_, _, info = model.generate_with_mask(..., enable_self_correction=True)
for i, h in enumerate(info['history']):
    print(f"Iter {i}: score={h['consistency_score']:.3f}")
```

### 可视化对齐矩阵
```python
from model.encoders.uni_alignment import UnifiedAlignmentModule

alignment = model.model.alignment_module
outputs = alignment(image_features, text_features, return_loss=True)

# 相似度矩阵 [B, B]
similarity = outputs['image_embeds'] @ outputs['text_embeds'].T
print(similarity)  # 对角线应该较大（正样本）
```

---

## 📊 性能优化

### 1. 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(...)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. DeepSpeed ZeRO
```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "zero_optimization": {
            "stage": 2
        }
    }
)
```

### 3. 梯度检查点
```python
# 在配置中启用
config = Med3DLISAConfig(
    gradient_checkpointing=True,  # 需要添加此参数
    ...
)
```

---

## ❓ 常见问题 (FAQ)

### Q1: 如何只使用部分 Stage？
A: 通过设置损失权重为 0 来禁用某个 Stage：
```python
config = Med3DLISAConfig(
    lambda_alignment=0.0,  # 禁用 Stage 1 alignment loss
    lambda_matching=0.0    # 禁用 Stage 4 matching loss
)
```

### Q2: 如何调整自我修正的敏感度？
A: 调整 `consistency_threshold`：
- 更高阈值 (0.8-0.9): 更严格，更多修正迭代
- 更低阈值 (0.5-0.6): 更宽松，更少修正迭代

### Q3: 内存不足怎么办？
A: 
1. 减小 batch size
2. 使用梯度累积
3. 启用 gradient checkpointing
4. 降低图像分辨率
5. 使用 DeepSpeed ZeRO-3

### Q4: 如何加速推理？
A:
1. 关闭 `enable_self_correction`
2. 使用 `torch.no_grad()`
3. 批量推理
4. 模型量化 (FP16/INT8)

---

## 📚 更多资源

- **完整文档**: [ARCHITECTURE_UPDATE.md](ARCHITECTURE_UPDATE.md)
- **测试脚本**: `test_full_integration.py`
- **训练脚本**: `train_net.py`
- **配置示例**: `config/med3d_lisa_full.yaml`

---

**版本**: v2.0  
**更新**: 2026-01-07
