"""
Med3D-MoE-Seg Model Package
直接使用复制的 MedPLIB 代码，无需动态导入
"""

# 直接从本地模块导入（已复制 MedPLIB 完整实现）
from .medplib_core.medplib_moe_llama import (
    MedPLIBMoELlamaModel,
    MedPLIBMoELlamaForCausalLM,
    MedPLIBMoELlamaConfig
)

from .medplib_core.llava_arch import (
    LlavaMetaModel,
    LlavaMetaForCausalLM
)

# 导入项目的核心模型
from .meta_arch.med3d_lisa import Med3DLISA, Med3DLISAConfig

# 标记所有组件都可用（因为已经内置）
HAS_MEDPLIB_MOE = True
HAS_MEDPLIB_ARCH = True

print("=" * 60)
print("Med3D-MoE-Seg Model:")
print("  ✓ MedPLIB MoE LLaMA (内置)")
print("  ✓ LLaVA 架构 (内置)")
print("  ✓ Med3D-LISA 核心模型")
print("=" * 60)

__all__ = [
    'Med3DLISA',
    'Med3DLISAConfig',
    'MedPLIBMoELlamaModel',
    'MedPLIBMoELlamaForCausalLM', 
    'MedPLIBMoELlamaConfig',
    'LlavaMetaModel',
    'LlavaMetaForCausalLM',
    'HAS_MEDPLIB_MOE',
    'HAS_MEDPLIB_ARCH',
]
