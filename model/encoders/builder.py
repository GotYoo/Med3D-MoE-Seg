"""
Vision Encoder Builder
"""

from .ct_clip_adapter import CTCLIPAdapter


def build_vision_encoder(config):
    """
    构建视觉编码器
    
    Args:
        config: 配置字典
    
    Returns:
        vision_encoder: 视觉编码器实例
    """
    encoder_type = config.get('vision_tower', 'ct_clip')
    
    if encoder_type == 'ct_clip':
        return CTCLIPAdapter(config)
    else:
        raise ValueError(f"Unknown vision encoder type: {encoder_type}")
