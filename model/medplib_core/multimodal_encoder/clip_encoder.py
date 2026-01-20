import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.vision_tower_name
            )
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name, low_cpu_mem_usage=True
            )
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True
            print(f"✓ Successfully loaded vision tower: {self.vision_tower_name}")
        except Exception as e:
            print(f"⚠ Failed to load vision tower '{self.vision_tower_name}': {e}")
            print(f"⚠ Using dummy vision tower instead")
            # 创建一个虚拟的 vision tower
            self.image_processor = None
            self.vision_tower = None
            self.is_loaded = False

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # 如果模型加载失败，返回零张量
        if not self.is_loaded or self.vision_tower is None:
            if type(images) is list:
                # 返回虚拟特征 [batch, num_patches, hidden_size]
                return [torch.zeros(1, 256, 768, device=images[0].device, dtype=images[0].dtype) for _ in images]
            else:
                batch_size = images.shape[0]
                return torch.zeros(batch_size, 256, 768, device=images.device, dtype=images.dtype)
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        torch.cuda.empty_cache()
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.dtype
        return torch.float32  # 默认返回 float32

    @property
    def device(self):
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.device
        return torch.device('cpu')  # 默认返回 CPU

    @property
    def config(self):
        if self.is_loaded and self.vision_tower is not None:
            return self.vision_tower.config
        elif hasattr(self, 'cfg_only'):
            return self.cfg_only
        else:
            # 返回默认配置
            from transformers import CLIPVisionConfig
            return CLIPVisionConfig()

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
    def freeze(self):
        """冻结 vision tower 参数"""
        if self.is_loaded and self.vision_tower is not None:
            self.vision_tower.requires_grad_(False)
            print(f"✓ Vision tower frozen")
        else:
            print(f"⚠ No vision tower to freeze")
