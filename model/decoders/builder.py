"""
Mask Decoder Builder
"""

from .sam_med3d_adapter import SAMMed3DAdapter


def build_mask_decoder(config):
    """
    构建掩码解码器
    
    Args:
        config: 配置字典
    
    Returns:
        mask_decoder: 掩码解码器实例
    """
    decoder_type = config.get('mask_decoder', 'sam_med3d')
    
    if decoder_type == 'sam_med3d':
        return SAMMed3DAdapter(config)
    else:
        raise ValueError(f"Unknown mask decoder type: {decoder_type}")
