from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    
    # 如果 vision_tower 为 None 或空字符串，返回 None（不使用视觉编码器）
    if vision_tower is None or vision_tower == "" or vision_tower == "null":
        return None
    
    # 转换为小写进行不区分大小写的匹配
    vision_tower_lower = vision_tower.lower()
    
    if (
        vision_tower_lower.startswith("openai")
        or vision_tower_lower.startswith("laion")
        or vision_tower_lower.startswith("microsoft")
        or "clip" in vision_tower_lower
        or "vit" in vision_tower_lower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
