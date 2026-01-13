"""
测试数据准备脚本
创建模拟数据并测试 prepare_data_split.py 的功能
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

def create_mock_nifti(filepath: Path, shape=(32, 64, 64)):
    """创建模拟的 NIfTI 文件（简化版，只创建空文件）"""
    # 实际应用中需要使用 nibabel 创建真实的 .nii.gz 文件
    # 这里为了测试，只创建空文件
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.touch()
    print(f"Created mock NIfTI: {filepath.name}")

def create_mock_report(filepath: Path, patient_id: str, scan_id: str):
    """创建模拟的报告 JSON 文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "findings": f"CT scan of patient {patient_id} shows findings consistent with normal anatomy.",
        "impression": "No significant abnormalities detected.",
        "report": f"Patient {patient_id}, Scan {scan_id}: Normal CT examination."
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created mock report: {filepath.name}")

def create_mock_dataset(base_dir: Path, n_patients: int = 10, scans_per_patient: int = 2):
    """
    创建模拟数据集
    
    Args:
        base_dir: 基础目录
        n_patients: 患者数量
        scans_per_patient: 每个患者的扫描数量
    """
    print("=" * 70)
    print(f"Creating mock dataset with {n_patients} patients")
    print("=" * 70)
    
    for i in range(n_patients):
        patient_id = f"LIDC-IDRI-{i+1:04d}"
        patient_dir = base_dir / patient_id
        
        for j in range(scans_per_patient):
            scan_id = f"scan{j+1:02d}"
            
            # 创建图像文件
            image_file = patient_dir / f"{patient_id}_{scan_id}.nii.gz"
            create_mock_nifti(image_file)
            
            # 创建掩码文件
            mask_file = patient_dir / f"{patient_id}_{scan_id}_mask.nii.gz"
            create_mock_nifti(mask_file)
            
            # 创建报告文件
            report_file = patient_dir / f"{patient_id}_{scan_id}_report.json"
            create_mock_report(report_file, patient_id, scan_id)
    
    print(f"\n✓ Created {n_patients * scans_per_patient} samples for {n_patients} patients")
    print(f"✓ Data directory: {base_dir}")

def test_data_split():
    """测试数据划分功能"""
    print("\n" + "=" * 70)
    print("Testing Data Split Functionality")
    print("=" * 70)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "mock_data"
        output_dir = temp_path / "splits"
        
        # 1. 创建模拟数据
        print("\n[Step 1] Creating mock dataset...")
        create_mock_dataset(data_dir, n_patients=10, scans_per_patient=2)
        
        # 2. 运行数据划分脚本
        print("\n[Step 2] Running data split script...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from scripts.prepare_data_split import (
            find_matching_files,
            split_patients,
            create_dataset_json,
            sanity_check,
            print_statistics
        )
        
        # 查找文件
        patient_data = find_matching_files(data_dir)
        print_statistics(patient_data)
        
        # 划分患者
        patient_ids = list(patient_data.keys())
        train_ids, val_ids, test_ids = split_patients(
            patient_ids,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            random_seed=42
        )
        
        # Sanity check
        sanity_check(train_ids, val_ids, test_ids)
        
        # 创建 JSON 文件
        output_dir.mkdir(parents=True, exist_ok=True)
        create_dataset_json(patient_data, train_ids, output_dir / 'train.json')
        create_dataset_json(patient_data, val_ids, output_dir / 'val.json')
        create_dataset_json(patient_data, test_ids, output_dir / 'test.json')
        
        # 3. 验证输出
        print("\n[Step 3] Validating outputs...")
        for split_name in ['train', 'val', 'test']:
            json_path = output_dir / f'{split_name}.json'
            assert json_path.exists(), f"{split_name}.json not created"
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n{split_name.upper()} set:")
            print(f"  - Samples: {len(data)}")
            print(f"  - Patients: {len(set(item['patient_id'] for item in data))}")
            
            # 检查数据格式
            if len(data) > 0:
                sample = data[0]
                required_keys = ['patient_id', 'image_path', 'mask_path', 'text_report']
                for key in required_keys:
                    assert key in sample, f"Missing key '{key}' in {split_name} sample"
                
                print(f"  - Sample keys: {list(sample.keys())}")
                print(f"  - Text report length: {len(sample['text_report'])} chars")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)

if __name__ == '__main__':
    test_data_split()
