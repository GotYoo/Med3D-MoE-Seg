# 数据准备指南

## 📋 概述

在训练 Med3D-LISA 模型之前，需要准备和划分训练数据。`prepare_data_split.py` 脚本实现了 **Patient-wise Split**，确保同一患者的所有扫描只出现在一个数据集（训练/验证/测试）中。

---

## 📁 输入数据格式

### 目录结构示例
```
data_root/
├── LIDC-IDRI-0001/
│   ├── LIDC-IDRI-0001_scan01.nii.gz          # CT 图像
│   ├── LIDC-IDRI-0001_scan01_mask.nii.gz     # 分割掩码
│   ├── LIDC-IDRI-0001_scan01_report.json     # 放射学报告
│   ├── LIDC-IDRI-0001_scan02.nii.gz
│   ├── LIDC-IDRI-0001_scan02_mask.nii.gz
│   └── LIDC-IDRI-0001_scan02_report.json
├── LIDC-IDRI-0002/
│   ├── LIDC-IDRI-0002_scan01.nii.gz
│   ├── LIDC-IDRI-0002_scan01_mask.nii.gz
│   └── LIDC-IDRI-0002_scan01_report.json
└── ...
```

### 文件命名规范

**支持的命名格式**:
- `PatientID_ScanID.nii.gz` (推荐)
- `LIDC-IDRI-0001_scan01.nii.gz`
- `Patient001_CT.nii.gz`

**掩码文件**: 在原文件名后添加 `_mask` 或 `_seg`
- `PatientID_ScanID_mask.nii.gz`
- `PatientID_ScanID_seg.nii.gz`

**报告文件**: JSON 格式，包含以下字段之一
- `report`: 完整报告文本
- `findings`: 发现部分
- `impression`: 印象部分
- `text`: 文本内容

报告 JSON 示例:
```json
{
  "patient_id": "LIDC-IDRI-0001",
  "scan_id": "scan01",
  "findings": "The CT scan shows a 5mm nodule in the right upper lobe.",
  "impression": "Small pulmonary nodule, recommend follow-up.",
  "report": "Complete report text here..."
}
```

---

## 🚀 快速开始

### 方法 1: 使用 Bash 脚本（推荐）

```bash
# 1. 编辑 scripts/prepare_data.sh，设置数据路径
vim scripts/prepare_data.sh

# 修改这些变量:
DATA_DIR="/path/to/your/raw_data"
OUTPUT_DIR="/path/to/output/splits"

# 2. 运行脚本
bash scripts/prepare_data.sh
```

### 方法 2: 直接运行 Python 脚本

```bash
python scripts/prepare_data_split.py \
    --data_dir /home/wuhanqing/processed_lidc_data \
    --output_dir /home/wuhanqing/Med3D-MoE-Seg/data/splits \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --random_seed 42
```

---

## 📊 输出格式

### 生成的文件
```
output_dir/
├── train.json    # 训练集
├── val.json      # 验证集
└── test.json     # 测试集
```

### JSON 格式
```json
[
  {
    "patient_id": "LIDC-IDRI-0001",
    "image_path": "/absolute/path/to/LIDC-IDRI-0001_scan01.nii.gz",
    "mask_path": "/absolute/path/to/LIDC-IDRI-0001_scan01_mask.nii.gz",
    "report_path": "/absolute/path/to/LIDC-IDRI-0001_scan01_report.json",
    "text_report": "Complete report text content..."
  },
  ...
]
```

---

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | **必需** | 原始数据根目录 |
| `--output_dir` | str | **必需** | 输出 JSON 文件目录 |
| `--train_ratio` | float | 0.7 | 训练集比例 (70%) |
| `--val_ratio` | float | 0.1 | 验证集比例 (10%) |
| `--test_ratio` | float | 0.2 | 测试集比例 (20%) |
| `--random_seed` | int | 42 | 随机种子（保证可重复性） |
| `--image_pattern` | str | `*.nii.gz` | 图像文件匹配模式 |

---

## 🔍 Patient-wise Split 原理

### 为什么需要 Patient-wise Split？

**问题**: 如果随机划分样本（Sample-wise Split）：
- 同一患者的多次扫描可能分布在训练集和测试集
- 导致**数据泄露**：模型可能记住患者特征而非疾病特征
- 过度乐观的性能评估

**解决方案**: Patient-wise Split
```
Patient-001:
  - scan1 → Train
  - scan2 → Train  ✅ 所有扫描都在训练集

Patient-002:
  - scan1 → Test
  - scan2 → Test   ✅ 所有扫描都在测试集
```

### 实现逻辑

1. **解析 Patient ID**: 从文件名提取患者标识
   ```python
   "LIDC-IDRI-0001_scan01.nii.gz" → "LIDC-IDRI-0001"
   ```

2. **按患者分组**: 收集每个患者的所有扫描
   ```python
   {
     "LIDC-IDRI-0001": [scan1, scan2, scan3],
     "LIDC-IDRI-0002": [scan1],
     ...
   }
   ```

3. **随机划分患者**: 按比例划分患者 ID 列表
   ```python
   all_patients → shuffle → [train_patients, val_patients, test_patients]
   ```

4. **Sanity Check**: 验证患者 ID 没有交集
   ```python
   assert train_set ∩ val_set == ∅
   assert train_set ∩ test_set == ∅
   assert val_set ∩ test_set == ∅
   ```

---

## 📈 示例输出

```
======================================================================
Med3D-MoE-Seg Data Preparation
======================================================================
Data directory: /home/wuhanqing/processed_lidc_data
Output directory: /home/wuhanqing/Med3D-MoE-Seg/data/splits
Split ratios: Train=0.7, Val=0.1, Test=0.2
Random seed: 42

[Step 1] Finding matching files...
Found 240 image files

======================================================================
Data Statistics
======================================================================
Total patients: 120
Total samples: 240
  - With mask: 240 (100.0%)
  - With report: 220 (91.7%)
  - Complete (image+mask+report): 220 (91.7%)
Average samples per patient: 2.00
======================================================================

[Step 2] Splitting patients into train/val/test sets...

======================================================================
Sanity Check: Patient ID Overlap
======================================================================
Train set: 84 patients
Val set: 12 patients
Test set: 24 patients

Overlap Check:
  Train ∩ Val: 0 patients
  Train ∩ Test: 0 patients
  Val ∩ Test: 0 patients

✅ Sanity check passed! No patient ID overlaps between splits.
======================================================================

[Step 3] Creating JSON files...
Created train.json with 168 samples from 84 patients
Created val.json with 24 samples from 12 patients
Created test.json with 48 samples from 24 patients

======================================================================
Data preparation completed successfully!
======================================================================
Output files:
  - /home/wuhanqing/Med3D-MoE-Seg/data/splits/train.json
  - /home/wuhanqing/Med3D-MoE-Seg/data/splits/val.json
  - /home/wuhanqing/Med3D-MoE-Seg/data/splits/test.json
======================================================================
```

---

## 🛠️ 高级用法

### 自定义文件匹配模式

如果数据使用不同的命名规范：

```bash
python scripts/prepare_data_split.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --image_pattern "*.nii"  # 匹配 .nii 而不是 .nii.gz
```

### 修改 Patient ID 解析逻辑

编辑 `scripts/prepare_data_split.py` 中的 `parse_patient_id()` 函数：

```python
def parse_patient_id(filename: str) -> str:
    # 自定义解析逻辑
    # 例如：使用正则表达式
    import re
    match = re.search(r'P(\d+)', filename)
    if match:
        return f"Patient{match.group(1)}"
    return filename
```

### 处理缺失数据

脚本会自动处理：
- ✅ 图像有，掩码缺失 → 记录但 `mask_path = null`
- ✅ 图像有，报告缺失 → 记录但 `text_report = ""`
- ❌ 没有图像文件 → 跳过

检查缺失率：
```bash
python -c "
import json
data = json.load(open('data/splits/train.json'))
n_total = len(data)
n_no_mask = sum(1 for item in data if item['mask_path'] is None)
n_no_report = sum(1 for item in data if not item['text_report'])
print(f'Missing masks: {n_no_mask}/{n_total} ({n_no_mask/n_total*100:.1f}%)')
print(f'Missing reports: {n_no_report}/{n_total} ({n_no_report/n_total*100:.1f}%)')
"
```

---

## ✅ 验证数据集

### 检查文件完整性

```bash
python -c "
import json
from pathlib import Path

for split in ['train', 'val', 'test']:
    data = json.load(open(f'data/splits/{split}.json'))
    print(f'{split.upper()} set:')
    
    for item in data:
        # 检查文件是否存在
        if not Path(item['image_path']).exists():
            print(f'  ❌ Missing image: {item[\"image_path\"]}')
        if item['mask_path'] and not Path(item['mask_path']).exists():
            print(f'  ❌ Missing mask: {item[\"mask_path\"]}')
    
    print(f'  ✅ All files exist')
"
```

### 可视化数据分布

```python
import json
import matplotlib.pyplot as plt

# 加载数据
splits = {}
for split in ['train', 'val', 'test']:
    with open(f'data/splits/{split}.json') as f:
        splits[split] = json.load(f)

# 统计患者数
patient_counts = {
    split: len(set(item['patient_id'] for item in data))
    for split, data in splits.items()
}

# 绘图
plt.bar(patient_counts.keys(), patient_counts.values())
plt.xlabel('Split')
plt.ylabel('Number of Patients')
plt.title('Patient Distribution Across Splits')
plt.savefig('data_distribution.png')
```

---

## 🐛 常见问题

### Q1: "No patient data found"
**原因**: 数据目录不正确或文件命名不符合规范

**解决**:
1. 检查 `--data_dir` 路径是否正确
2. 确认文件扩展名（`.nii.gz` vs `.nii`）
3. 使用 `--image_pattern` 自定义匹配模式

### Q2: Patient ID 解析不正确
**原因**: 文件命名格式特殊

**解决**: 修改 `parse_patient_id()` 函数以适配您的命名规范

### Q3: 报告文本为空
**原因**: JSON 字段名不匹配

**解决**: 检查报告 JSON 的字段名，修改 `load_report()` 函数

### Q4: 比例不精确
**原因**: 患者数量较少时，整数划分导致比例偏差

**示例**: 10 个患者，7:1:2 → 实际 7:1:2 (70%:10%:20%) ✅
         11 个患者，7:1:2 → 实际 7:1:3 (64%:9%:27%) ❌

**解决**: 如果患者数量 < 50，考虑调整比例或接受偏差

---

## 📝 下一步

数据准备完成后：

1. **检查生成的 JSON 文件**
   ```bash
   head -20 data/splits/train.json
   ```

2. **更新训练配置**
   ```yaml
   # config/med3d_lisa_full.yaml
   data:
     train_data: data/splits/train.json
     val_data: data/splits/val.json
     test_data: data/splits/test.json
   ```

3. **开始训练**
   ```bash
   bash scripts/train_ds.sh
   ```

---

**脚本位置**: `scripts/prepare_data_split.py`  
**示例脚本**: `scripts/prepare_data.sh`  
**测试脚本**: `scripts/test_data_split.py`
