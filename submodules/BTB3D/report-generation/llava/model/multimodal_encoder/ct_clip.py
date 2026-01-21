import torch
import torch.nn as nn
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformer_maskgit import CTViT

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        #self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)


        self.vision_tower = CTViT(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        ).cuda().half() 
        
        # 【Fix】使用 self.vision_tower_name (即你 config 里配的 checkpoint 路径)
        if self.vision_tower_name and os.path.exists(self.vision_tower_name):
            print(f"Loading CTViT weights from: {self.vision_tower_name}")
            self.vision_tower.load(self.vision_tower_name)
        else:
            print(f"Warning: Checkpoint not found at {self.vision_tower_name}, using random init.")

        #self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, images):
        images = images.unsqueeze(1)
        # print(images.shape)
        # print(images.max())
        # print(images.min())
        image_features = self.vision_tower(images)
        a , b = image_features.shape[0], image_features.shape[-1]
        image_features = image_features.view(a, -1, b)
        # print("test_size")
        # print(image_features.shape)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


