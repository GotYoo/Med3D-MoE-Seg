import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import logging
import os

logger = logging.getLogger(__name__)

class AlignmentModel(nn.Module):
    """
    Stage 1 Unified Alignment Model (A100 Version)
    Architecture:
    1. Student: ViT-Large (Unified Vision Tower) - 负责处理输入并学习文本对齐。
    2. Teacher: Frozen 3D Encoder (CT-CLIP/BTB3D) - 负责提供 3D 结构特征供 Student 蒸馏。
    3. Alignment: Global (Volume-level) + Local (Region/Pixel-level).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        
        # ==================== 1. Student: Unified Vision Tower ====================
        vision_cfg = model_cfg.get('unified_vision_encoder', {})
        self.vision_tower_name = vision_cfg.get('vision_tower', 'openai/clip-vit-large-patch14-336')
        logger.info(f"Building Student Vision Tower: {self.vision_tower_name}")
        
        # 加载 HuggingFace CLIP ViT
        self.student_model = AutoModel.from_pretrained(self.vision_tower_name)
        if hasattr(self.student_model, "vision_model"):
            self.student_model = self.student_model.vision_model
            
        # 冻结/解冻 Student
        if vision_cfg.get('freeze_vision_tower', False):
            self.student_model.eval()
            for p in self.student_model.parameters():
                p.requires_grad = False
        else:
            self.student_model.train()
            # 可以选择只解冻最后几层
            if vision_cfg.get('unfreeze_layers', -1) > 0:
                # 先冻结所有
                for p in self.student_model.parameters():
                    p.requires_grad = False
                # 解冻最后 N 层 encoder.layers
                layers = self.student_model.encoder.layers
                n_layers = len(layers)
                for i in range(n_layers - vision_cfg['unfreeze_layers'], n_layers):
                    for p in layers[i].parameters():
                        p.requires_grad = True
        
        # 维度投影
        student_dim = self.student_model.config.hidden_size
        embed_dim = model_cfg.get('alignment', {}).get('embed_dim', 768)
        
        self.vision_global_proj = nn.Linear(student_dim, embed_dim)
        self.vision_local_proj = nn.Linear(student_dim, embed_dim)

        # ==================== 2. Teacher: External 3D Encoder (Frozen) ====================
        teacher_cfg = model_cfg.get('external_3d_encoder', {})
        self.use_distillation = teacher_cfg.get('enabled', False)
        self.teacher_model = None
        
        if self.use_distillation:
            logger.info("Building Frozen Teacher 3D Encoder for Distillation...")
            
            try:
                # 1. 导入你刚才发给我的 Adapter
                from model.encoders.ct_clip_adapter import CTCLIPVisionTower
                
                # 2. 构造一个简单的 Config 对象 (Adapter 需要这个参数)
                class SimpleAdapterConfig:
                    mm_vision_select_layer = -2
                    mm_vision_select_feature = 'patch'
                
                adapter_config = SimpleAdapterConfig()
                
                # 3. 获取权重路径
                ckpt_path = teacher_cfg.get('checkpoint')
                if not ckpt_path:
                    logger.warning("No checkpoint path provided for Teacher, will use random init or fallback!")
                
                # 4. 实例化 Teacher
                # 注意：这里直接加载你的 3D 权重
                self.teacher_model = CTCLIPVisionTower(
                    vision_tower_path=ckpt_path,
                    config=adapter_config,
                    delay_load=False
                )
                
                logger.info(f"Teacher Model Loaded: {type(self.teacher_model)}")
                
                # 5. 彻底冻结 Teacher
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                
                # 6. 获取 Teacher 输出维度 (动态获取，不再写死 512)
                # Adapter 代码里有 @property hidden_size
                teacher_dim = self.teacher_model.hidden_size
                logger.info(f"Teacher Hidden Size: {teacher_dim}")
                
                # 7. 建立投影层 (Teacher Dim -> Student Dim)
                self.distill_proj = nn.Linear(teacher_dim, embed_dim)

            except Exception as e:
                # 把 Exception 转为 string 打印出来
                logger.error(f"Failed to load CTCLIPVisionTower: {str(e)}") # <--- 加 str(e)
                import traceback
                traceback.print_exc() # <--- 打印堆栈
                
                logger.warning("Disabling distillation due to Teacher loading failure.")
                self.use_distillation = False
                self.teacher_model = None

        # ==================== 3. Text Encoder (BioBERT) ====================
        text_cfg = model_cfg.get('text_encoder', {})
        self.text_model_name = text_cfg.get('model_name', 'dmis-lab/biobert-v1.1')
        logger.info(f"Building Text Encoder: {self.text_model_name}")
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        
        if text_cfg.get('freeze_biobert', False):
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad = False
        
        text_dim = self.text_model.config.hidden_size
        self.text_global_proj = nn.Linear(text_dim, embed_dim)
        self.text_local_proj = nn.Linear(text_dim, embed_dim)

        # 温度系数 (可学习)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, ct_volume, text_inputs, return_loss=True, **kwargs):
        """
        ct_volume: [B, 1, D, H, W]
        text_inputs: dict (input_ids, attention_mask)
        """
        # --- A. 处理 Student (Unified ViT) ---
        # Reshape 3D -> 2D Sequence: [B, 1, D, H, W] -> [B*D, 3, H, W] (3 channels for ViT)
        b, c, d, h, w = ct_volume.shape
        
        # 1. 调整维度
        x = ct_volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        # 2. 如果是单通道 CT，复制为 3 通道适配 ViT
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 3. 【关键修复】自动 Resize 到模型需要的尺寸 (例如 336)
        # 必须先 Resize 再送入模型，否则 96x96 必挂
        target_size = 336 
        # 尝试从配置获取真实尺寸
        if hasattr(self.student_model.config, 'vision_config'):
             target_size = self.student_model.config.vision_config.image_size
        elif hasattr(self.student_model.config, 'image_size'):
             target_size = self.student_model.config.image_size
             
        if x.shape[-1] != target_size:
            # 双线性插值缩放
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # 3. ViT Forward
        student_out = self.student_model(pixel_values=x, output_hidden_states=True)
        
        # 4. 提取特征
        # Global: [B*D, Dim] -> [B, D, Dim] -> Mean -> [B, Dim]
        s_cls = student_out.pooler_output.view(b, d, -1).mean(dim=1)
        s_global = self.vision_global_proj(s_cls)
        s_global = F.normalize(s_global, dim=-1)
        
        # Local: [B*D, N_patches, Dim] -> [B, D, N_patches, Dim] -> Flatten -> [B, L_vis, Dim]
        s_patch = student_out.last_hidden_state[:, 1:, :] # Skip CLS
        s_local_raw = s_patch.view(b, d, -1, s_patch.size(-1)).reshape(b, -1, s_patch.size(-1))
        s_local = self.vision_local_proj(s_local_raw)
        s_local = F.normalize(s_local, dim=-1)

        # --- B. 处理 Teacher (3D Encoder) ---
        t_global_feat = None
        # --- B. 处理 Teacher (CT-CLIP) ---
        if self.use_distillation and self.teacher_model is not None:
            # 1. 强制关闭 Autocast (配合第三方库要求)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                # 确保 Teacher 权重为 FP32
                self.teacher_model.float() 
                
                target_h, target_w = 480, 480
                target_d = 11
                
                # 2. 准备 FP32 输入 + 3D 插值
                t_input = ct_volume.to(dtype=torch.float32)
                t_input = F.interpolate(
                    t_input, 
                    size=(target_d, target_h, target_w), 
                    mode='trilinear', 
                    align_corners=False
                )
                
                # 3. 前向传播
                t_out = self.teacher_model(t_input)
                
                if isinstance(t_out, tuple):
                    t_out = t_out[0]
                
                # 【Fix Warning】如果输出形状是 [B, 1, Dim]，压缩掉中间的 1
                if t_out.dim() == 3 and t_out.shape[1] == 1:
                    t_out = t_out.squeeze(1)  # 变成 [B, Dim]
                
                # 4. 转回 BF16/FP16 匹配 Student
                t_out = t_out.to(dtype=ct_volume.dtype)
                
            # 投影到对齐空间
            t_global_feat = self.distill_proj(t_out)
            t_global_feat = F.normalize(t_global_feat, dim=-1)

        # --- C. 处理 Text ---
        txt_out = self.text_model(**text_inputs)
        
        # Global
        txt_cls = txt_out.last_hidden_state[:, 0, :]
        txt_global = self.text_global_proj(txt_cls)
        txt_global = F.normalize(txt_global, dim=-1)
        
        # Local
        txt_word = txt_out.last_hidden_state[:, 1:, :]
        txt_local = self.text_local_proj(txt_word)
        txt_local = F.normalize(txt_local, dim=-1)

        if return_loss:
            # 获取可能存在的 region_masks
            region_masks = kwargs.get('region_masks', None)
            
            return self.compute_losses(
                s_global, txt_global,
                s_local, txt_local,
                t_global_feat,
                text_inputs['attention_mask'][:, 1:], # 去掉 CLS 的 mask
                region_masks
            )
        
        return {"v_global": s_global, "t_global": txt_global}

    def compute_losses(self, v_pred, t_pred, v_local, t_local, v_teacher, text_mask, region_masks=None):
        loss_dict = {}
        weights = self.config.get('training', {}).get('loss_weights', {})

        # 1. Global Contrastive Loss
        logit_scale = self.logit_scale.exp()
        logits_img = logit_scale * v_pred @ t_pred.t()
        logits_txt = logit_scale * t_pred @ v_pred.t()
        labels = torch.arange(len(v_pred), device=v_pred.device)
        loss_global = (F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_txt, labels)) / 2
        
        # 2. 3D Distillation Loss
        loss_distill = torch.tensor(0.0, device=v_pred.device)
        if v_teacher is not None:
            # MSE Loss 让 Student 模仿 Teacher 的分布
            loss_distill = F.mse_loss(v_pred, v_teacher)

        # 3. Local/Region Loss (自适应)
        loss_local = torch.tensor(0.0, device=v_pred.device)
        
        # 如果提供了 Region Mask (RadGenome 强监督)
        if region_masks is not None and region_masks.sum() > 0:
            # 这里简化处理：如果没有实现复杂的 Region Pooling，
            # 可以回退到 GLoRIA，或者只对有 Mask 的样本计算
            # 暂时使用 GLoRIA 兜底，防止代码报错
            pass 
            
        # 默认使用 GLoRIA (Attention-based 弱监督)
        # 适合 LIDC 或 Mask 处理尚未完善的情况
        sim_map = torch.bmm(t_local, v_local.transpose(1, 2)) # [B, L_txt, L_img]
        max_sim, _ = sim_map.max(dim=-1) # [B, L_txt]
        
        if text_mask is not None:
            mask = text_mask.float()
            loss_local = -(max_sim * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_local = -max_sim.mean()

        # 总损失
        total_loss = (weights.get('global_contrastive_loss', 1.0) * loss_global +
                      weights.get('local_contrastive_loss', 0.5) * loss_local +
                      weights.get('distillation_loss', 2.0) * loss_distill)

        loss_dict.update({
            'loss': total_loss,
            'loss_global': loss_global,
            'loss_local': loss_local,
            'loss_distill': loss_distill
        })
        
        return loss_dict
    def get_text_tokenizer(self):
        """
        辅助方法：获取与当前 Text Encoder 匹配的 Tokenizer
        """
        from transformers import AutoTokenizer
        # 使用初始化时保存的 model_name (默认为 biobert)
        tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        
        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
