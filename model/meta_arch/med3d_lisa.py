"""
Med3D-LISA: 3D Medical Image Segmentation with Language Instruction
整合 CT-CLIP、MoE-LLaMA 和 SAM-Med3D 的核心架构
参考 LISA (Language Instructed Segmentation Assistant) 论文

完整 4-Stage 流程：
Stage 1: Multi-Modal Alignment (BioBERT + CT-CLIP + Contrastive Learning)
Stage 2: RAG Knowledge Retrieval (Medical Knowledge Base)
Stage 3: MoE LLM Reasoning (8-Expert MoE-LLaMA)
Stage 4: Self-Correction Loop (Consistency Checking + Iterative Refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Dict, List

from ..medplib_core.medplib_moe_llama import MedPLibMoELlamaForCausalLM
from ..medplib_core.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..encoders.ct_clip_adapter import CTCLIPVisionTower
from ..encoders.biobert_encoder import BioBERTEncoder
from ..encoders.uni_alignment import UnifiedAlignmentModule
from ..rag.retriever import MedicalKnowledgeRetriever
from ..correction.consistency import ConsistencyChecker
from ..decoders.sam_med3d_adapter import SAMMed3DMaskDecoder


class Med3DLISAConfig(PretrainedConfig):
    """Med3D-LISA 配置类，继承 PretrainedConfig 以兼容 transformers 接口"""

    model_type = "med3d_lisa"

    def __init__(self, **kwargs):
        # LLM 配置
        self.hidden_size = kwargs.get('hidden_size', 4096)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 32)
        self.num_attention_heads = kwargs.get('num_attention_heads', 32)
        # LLaMA requires num_key_value_heads; default to attention heads if not provided
        self.num_key_value_heads = kwargs.get('num_key_value_heads', self.num_attention_heads)
        self.intermediate_size = kwargs.get('intermediate_size', 11008)
        # Core LLaMA config fields expected by transformers
        self.attention_dropout = kwargs.get('attention_dropout', 0.0)
        self.attention_bias = kwargs.get('attention_bias', False)
        self.hidden_act = kwargs.get('hidden_act', 'silu')
        self.mlp_bias = kwargs.get('mlp_bias', False)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-5)
        self.use_cache = kwargs.get('use_cache', True)
        self.pretraining_tp = kwargs.get('pretraining_tp', 1)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        self.rope_scaling = kwargs.get('rope_scaling', None)
        
        # MoE 配置
        self.num_experts = kwargs.get('num_experts', 8)
        self.num_experts_per_tok = kwargs.get('num_experts_per_tok', 2)
        
        # Vision 配置
        self.vision_tower = kwargs.get('vision_tower', 'ct_clip')
        self.vision_hidden_size = kwargs.get('vision_hidden_size', 768)
        self.mm_projector_type = kwargs.get('mm_projector_type', 'mlp2x_gelu')
        
        # 分割配置
        self.seg_token_idx = kwargs.get('seg_token_idx', 32000)
        self.image_token_index = kwargs.get('image_token_index', -200)
        
        # SAM 配置
        self.sam_type = kwargs.get('sam_type', 'vit_b_ori')
        self.sam_checkpoint = kwargs.get('sam_checkpoint', None)
        
        # ===== NEW: Stage 1-4 配置 =====
        # Stage 1: Multi-Modal Alignment
        self.biobert_model = kwargs.get('biobert_model', 'dmis-lab/biobert-v1.1')
        self.biobert_freeze_layers = kwargs.get('biobert_freeze_layers', 8)
        self.text_hidden_size = kwargs.get('text_hidden_size', 768)
        self.latent_dim = kwargs.get('latent_dim', 512)
        self.alignment_temperature = kwargs.get('alignment_temperature', 0.07)
        
        # Stage 2: RAG
        self.rag_top_k = kwargs.get('rag_top_k', 3)
        self.rag_knowledge_dim = kwargs.get('rag_knowledge_dim', 512)
        self.rag_num_entries = kwargs.get('rag_num_entries', 1000)
        self.rag_injection_position = kwargs.get('rag_injection_position', 'prepend')
        
        # Stage 4: Self-Correction
        self.consistency_mask_channels = kwargs.get('consistency_mask_channels', 256)
        self.consistency_embed_dim = kwargs.get('consistency_embed_dim', 512)
        self.consistency_num_heads = kwargs.get('consistency_num_heads', 8)
        self.consistency_threshold = kwargs.get('consistency_threshold', 0.7)
        self.max_correction_iterations = kwargs.get('max_correction_iterations', 3)
        
        # 损失权重
        self.lambda_alignment = kwargs.get('lambda_alignment', 0.1)
        self.lambda_dice = kwargs.get('lambda_dice', 1.0)
        self.lambda_matching = kwargs.get('lambda_matching', 0.5)
        self.seg_loss_weight = kwargs.get('seg_loss_weight', 1.0)

        # 调用父类初始化，设置 vocab_size 等必要字段
        super().__init__(
            pad_token_id=kwargs.get('pad_token_id', 0),
            bos_token_id=kwargs.get('bos_token_id', 1),
            eos_token_id=kwargs.get('eos_token_id', 2),
            vocab_size=kwargs.get('vocab_size', 32000),
        )


class Med3DLISAModel(LlavaMetaModel, MedPLibMoELlamaForCausalLM):
    """
    Med3D-LISA 模型主体 (完整 4-Stage 版本)
    继承 LlavaMetaModel（多模态融合）和 MedPLibMoELlamaForCausalLM（MoE LLM）
    
    Architecture:
    Stage 1: CT-CLIP (Image) + BioBERT (Text) → Unified Alignment
    Stage 2: RAG → Medical Knowledge Retrieval → Context Injection
    Stage 3: MoE-LLaMA → Language Understanding & Reasoning
    Stage 4: SAM-Med3D → Mask Generation + Consistency Checking
    """
    
    def __init__(self, config):
        super(Med3DLISAModel, self).__init__(config)
        
        self.config = config
        
        # ========== Stage 1: Multi-Modal Encoding & Alignment ==========
        # 1.1 Vision Tower (CT-CLIP)
        self.vision_tower = CTCLIPVisionTower(
            vision_tower_path=config.vision_tower,
            config=config,
            delay_load=False
        )
        
        # 1.2 Text Encoder (BioBERT)
        self.text_encoder = BioBERTEncoder(
            model_name=getattr(config, 'biobert_model', 'dmis-lab/biobert-v1.1'),
            freeze_layers=getattr(config, 'biobert_freeze_layers', 8)
        )
        
        # 1.3 Unified Alignment Module
        self.alignment_module = UnifiedAlignmentModule(
            image_dim=self.vision_tower.hidden_size,  # CT-CLIP: 512
            text_dim=self.text_encoder.hidden_size,   # BioBERT: 768
            latent_dim=getattr(config, 'latent_dim', 512),
            temperature=getattr(config, 'alignment_temperature', 0.07)
        )
        
        # ========== Stage 2: RAG Knowledge Retrieval ==========
        self.rag_retriever = MedicalKnowledgeRetriever(
            knowledge_dim=getattr(config, 'rag_knowledge_dim', 512),
            llm_hidden_size=config.hidden_size,
            top_k=getattr(config, 'rag_top_k', 3),
            num_knowledge_entries=getattr(config, 'rag_num_entries', 1000)
        )
        
        # ========== Stage 3: Multimodal Projector (for LLM) ==========
        self._init_mm_projector(config)
        
        # ========== Stage 4: Segmentation & Self-Correction ==========
        # 4.1 Segmentation Decoder (SAM-Med3D)
        self.seg_decoder = SAMMed3DMaskDecoder(
            model_type=getattr(config, 'sam_type', 'vit_b_ori'),
            checkpoint_path=getattr(config, 'sam_checkpoint', None),
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            llm_hidden_size=config.hidden_size
        )
        
        # 4.2 Consistency Checker
        self.consistency_checker = ConsistencyChecker(
            mask_channels=getattr(config, 'consistency_mask_channels', 256),
            text_hidden_size=config.hidden_size,
            embed_dim=getattr(config, 'consistency_embed_dim', 512),
            num_heads=getattr(config, 'consistency_num_heads', 8)
        )
        
        # 分割 token 索引
        self.seg_token_idx = getattr(config, 'seg_token_idx', 32000)
    
    def _init_mm_projector(self, config):
        """
        初始化多模态投影层
        将 Vision Tower 的输出投影到 LLM 的隐藏空间
        """
        projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
        
        if projector_type == 'linear':
            self.mm_projector = nn.Linear(
                self.vision_tower.hidden_size,
                config.hidden_size
            )
        elif projector_type == 'mlp2x_gelu':
            # 2层 MLP，中间使用 GELU 激活
            self.mm_projector = nn.Sequential(
                nn.Linear(self.vision_tower.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
    
    def get_vision_tower(self):
        """获取 Vision Tower"""
        return self.vision_tower
    
    def get_model(self):
        """获取 LLM 模型"""
        return self.model


class Med3DLISA_Full(PreTrainedModel, LlavaMetaForCausalLM):
    """
    Med3D-LISA: 完整的端到端模型（4-Stage 版本）
    
    完整流程：
    【Training】
    Stage 1: CT-CLIP + BioBERT → Alignment Loss (Contrastive)
    Stage 2: Query → RAG Retrieval → Knowledge Context
    Stage 3: [Knowledge] + [Image] + [Instruction] → MoE-LLaMA → Report + Seg Token
    Stage 4: Mask Generation + Consistency Check → Matching Loss
    Total Loss = LM Loss + λ₁·Dice Loss + λ₂·Alignment Loss + λ₃·Matching Loss
    
    【Inference】
    Draft Generation → Consistency Check → Iterative Refinement (Self-Correction Loop)
    """
    
    config_class = Med3DLISAConfig
    
    def __init__(self, config):
        super(Med3DLISA_Full, self).__init__(config)
        
        # 初始化主模型
        self.model = Med3DLISAModel(config)
        
        # LM head（用于文本生成）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 后初始化
        self.post_init()
    
    def get_model(self):
        """获取主模型"""
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        clinical_reports: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        masks_gt: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Dict:
        """
        前向传播（完整 4-Stage 流程）
        
        Args:
            input_ids: [B, L] 文本 token IDs
            attention_mask: [B, L] 注意力掩码
            images: [B, C, D, H, W] 3D 医学图像
            clinical_reports: Dict with 'input_ids' and 'attention_mask' for BioBERT
            labels: [B, L] 文本生成标签
            masks_gt: [B, 1, D, H, W] Ground truth 分割掩码
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
        
        Returns:
            dict: 包含 loss, logits, pred_masks, alignment_loss, matching_loss 等
        """
        
        # ========== Stage 1: Multi-Modal Alignment ==========
        alignment_loss = None
        image_features_aligned = None
        
        if images is not None:
            # 1.1 Extract image features (CT-CLIP)
            with torch.no_grad() if self.model.vision_tower.is_frozen else torch.enable_grad():
                image_features = self.model.vision_tower(images)  # [B, 512]
            
            # 1.2 Extract text features (BioBERT) if clinical reports provided
            if clinical_reports is not None:
                text_features = self.model.text_encoder(
                    input_ids=clinical_reports['input_ids'],
                    attention_mask=clinical_reports['attention_mask']
                )  # [B, 768]
                
                # 1.3 Unified Alignment
                alignment_outputs = self.model.alignment_module(
                    image_features,
                    text_features,
                    return_loss=True
                )
                image_features_aligned = alignment_outputs['image_embeds']  # [B, 512]
                alignment_loss = alignment_outputs['contrastive_loss']
            else:
                # 如果没有临床报告，只对图像特征进行投影
                image_features_aligned = self.model.alignment_module.image_projector(image_features)
        
        # ========== Stage 2: RAG Knowledge Retrieval ==========
        rag_context_embeds = None
        
        if image_features_aligned is not None:
            # 使用对齐后的图像特征作为查询
            rag_outputs = self.model.rag_retriever(
                query_embed=image_features_aligned,
                return_details=False
            )
            rag_context_embeds = rag_outputs['context_embed']  # [B, hidden_size]
        
        # ========== Stage 3: LLM Processing ==========
        # 3.1 准备多模态输入（将图像特征和 RAG 上下文融合到文本序列中）
        if images is not None:
            # 使用原始图像特征进行多模态投影（保持与 LlavaMetaModel 的兼容性）
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=labels,
                images=images
            )
            
            # 3.2 注入 RAG 上下文（在序列开头添加）
            if rag_context_embeds is not None:
                # 扩展维度 [B, hidden_size] → [B, 1, hidden_size]
                rag_embeds_expanded = rag_context_embeds.unsqueeze(1)
                
                # 拼接：[RAG Context] + [Original Sequence]
                inputs_embeds = torch.cat([rag_embeds_expanded, inputs_embeds], dim=1)
                
                # 更新 attention_mask
                rag_attention = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([rag_attention, attention_mask], dim=1)
                
                # 更新 labels（在开头添加 -100 忽略 RAG token）
                if labels is not None:
                    rag_labels = torch.full(
                        (labels.shape[0], 1),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([rag_labels, labels], dim=1)
        else:
            inputs_embeds = None
        
        # 3.3 LLM 前向传播（MoE-LLaMA 理解与推理）
        outputs = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        # 获取 hidden states
        hidden_states = outputs.hidden_states[-1]  # [B, L, hidden_size]
        
        # 3.4 文本生成 Logits
        lm_logits = self.lm_head(hidden_states)  # [B, L, vocab_size]
        
        # 3.5 计算文本生成 Loss
        lm_loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # ========== Stage 4: Segmentation & Self-Correction ==========
        pred_masks = None
        seg_loss = None
        matching_loss = None
        
        # 检测 input_ids 中是否包含 <SEG> token
        seg_token_mask = (input_ids == self.model.seg_token_idx) if input_ids is not None else None
        
        if seg_token_mask is not None and seg_token_mask.any():
            # 4.1 提取 <SEG> token 对应的 hidden states
            seg_hidden_states = self._extract_seg_hidden_states(hidden_states, seg_token_mask)
            
            if seg_hidden_states is not None and images is not None:
                # 4.2 生成图像 embeddings（用于 SAM）
                with torch.no_grad():
                    image_embeddings = self.model.seg_decoder.forward_image_features(images)
                
                # 4.3 使用 SAM-Med3D 生成掩码
                pred_masks = self.model.seg_decoder(
                    image_embeddings=image_embeddings,
                    seg_token_hidden_states=seg_hidden_states,
                    return_logits=True  # 返回 logits 用于计算损失
                )
                
                # 4.4 计算分割 Loss
                if masks_gt is not None:
                    seg_loss = self.compute_mask_loss(pred_masks, masks_gt)
                
                # 4.5 一致性检查（训练时使用 ground truth text embeddings）
                if labels is not None:
                    # 使用完整的 hidden states 作为文本表示
                    matching_outputs = self.model.consistency_checker(
                        mask_output=pred_masks,
                        text_embeds=hidden_states,
                        return_attention=False
                    )
                    consistency_score = matching_outputs['consistency_score']
                    
                    # 计算匹配损失（鼓励高一致性）
                    # 目标：正样本应该有高分数
                    target_consistency = torch.ones_like(consistency_score)
                    matching_loss = F.mse_loss(consistency_score, target_consistency)
        
        # ========== Total Loss Computation ==========
        total_loss = None
        if lm_loss is not None or seg_loss is not None or alignment_loss is not None or matching_loss is not None:
            total_loss = 0.0
            
            # Language modeling loss
            if lm_loss is not None:
                total_loss += lm_loss
            
            # Segmentation loss
            if seg_loss is not None:
                lambda_dice = getattr(self.config, 'lambda_dice', 1.0)
                total_loss += lambda_dice * seg_loss
            
            # Alignment loss
            if alignment_loss is not None:
                lambda_alignment = getattr(self.config, 'lambda_alignment', 0.1)
                total_loss += lambda_alignment * alignment_loss
            
            # Matching loss
            if matching_loss is not None:
                lambda_matching = getattr(self.config, 'lambda_matching', 0.5)
                total_loss += lambda_matching * matching_loss
        
        # ========== Return Results ==========
        if not return_dict:
            output = (lm_logits, pred_masks) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'seg_loss': seg_loss,
            'alignment_loss': alignment_loss,
            'matching_loss': matching_loss,
            'logits': lm_logits,
            'pred_masks': pred_masks,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'router_logits': outputs.router_logits if hasattr(outputs, 'router_logits') else None,
        }
    
    def _extract_seg_hidden_states(
        self,
        hidden_states: torch.Tensor,
        seg_token_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        提取 <SEG> token 对应的 hidden states
        
        Args:
            hidden_states: [B, L, hidden_size]
            seg_token_mask: [B, L] bool mask
        
        Returns:
            seg_hidden: [B, hidden_size] 或 None
        """
        if not seg_token_mask.any():
            return None
        
        batch_size = hidden_states.shape[0]
        seg_hidden_list = []
        
        for i in range(batch_size):
            # 找到第一个 <SEG> token 的位置
            seg_positions = torch.where(seg_token_mask[i])[0]
            if len(seg_positions) > 0:
                # 取第一个 <SEG> token 的 hidden state
                seg_hidden_list.append(hidden_states[i, seg_positions[0]])
            else:
                # 如果该样本没有 <SEG> token，使用零向量
                seg_hidden_list.append(torch.zeros(
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                ))
        
        return torch.stack(seg_hidden_list, dim=0)  # [B, hidden_size]
    
    def compute_mask_loss(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5
    ) -> torch.Tensor:
        """
        计算分割损失：BCE Loss + Dice Loss
        
        Args:
            pred_masks: [B, 1, D, H, W] 预测掩码 logits
            target_masks: [B, 1, D, H, W] Ground truth 掩码（0/1）
            bce_weight: BCE 损失权重
            dice_weight: Dice 损失权重
        
        Returns:
            loss: 总分割损失
        """
        # 1. BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks,
            target_masks,
            reduction='mean'
        )
        
        # 2. Dice Loss
        pred_probs = torch.sigmoid(pred_masks)
        dice_loss = self.dice_loss(pred_probs, target_masks)
        
        # 3. 组合损失
        total_loss = bce_weight * bce_loss + dice_weight * dice_loss
        
        return total_loss
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        计算 Dice Loss
        
        Args:
            pred: [B, 1, D, H, W] 预测概率
            target: [B, 1, D, H, W] Ground truth
            smooth: 平滑项，避免除零
        
        Returns:
            dice_loss: Dice 损失
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return dice_loss
    
    def generate_with_mask(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        clinical_reports: Optional[Dict[str, torch.Tensor]] = None,
        max_new_tokens: int = 512,
        enable_self_correction: bool = True,
        **generate_kwargs
    ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor], Dict]:
        """
        生成文本并预测分割掩码（带自我修正循环）
        
        Self-Correction Loop:
        1. Draft: 首次生成 Report + Mask
        2. Check: 使用 ConsistencyChecker 评估一致性
        3. Refine: 如果分数低于阈值，将 Draft 作为负面反馈再次生成
        4. Iterate: 重复直到达到阈值或最大迭代次数
        
        Args:
            input_ids: 输入 token IDs
            images: 3D 医学图像
            clinical_reports: 临床报告（用于 BioBERT）
            max_new_tokens: 最大生成 token 数
            enable_self_correction: 是否启用自我修正
            **generate_kwargs: 其他生成参数
        
        Returns:
            (generated_ids, pred_masks, correction_info)
        """
        device = input_ids.device
        
        # ========== Stage 1 & 2: Feature Extraction & RAG (一次性完成) ==========
        with torch.no_grad():
            # Extract and align features
            image_features = self.model.vision_tower(images)
            
            if clinical_reports is not None:
                text_features = self.model.text_encoder(
                    input_ids=clinical_reports['input_ids'],
                    attention_mask=clinical_reports['attention_mask']
                )
                alignment_outputs = self.model.alignment_module(
                    image_features, text_features, return_loss=False
                )
                image_features_aligned = alignment_outputs['image_embeds']
            else:
                image_features_aligned = self.model.alignment_module.image_projector(image_features)
            
            # RAG retrieval
            rag_outputs = self.model.rag_retriever(
                query_embed=image_features_aligned,
                return_details=False
            )
            rag_context_embeds = rag_outputs['context_embed']
        
        # ========== Self-Correction Loop ==========
        max_iterations = getattr(self.config, 'max_correction_iterations', 3)
        consistency_threshold = getattr(self.config, 'consistency_threshold', 0.7)
        
        correction_history = []
        best_result = None
        best_score = -1.0
        
        for iteration in range(max_iterations):
            # ========== Stage 3: Text Generation ==========
            # 准备输入（添加负面反馈提示，如果是修正迭代）
            current_input_ids = input_ids
            if iteration > 0 and best_result is not None:
                # 构建修正提示：
                # "Previous report was inconsistent with the mask. Please refine: [previous_report]"
                refinement_prompt = self._build_refinement_prompt(
                    best_result['generated_ids'],
                    best_result['consistency_score']
                )
                current_input_ids = refinement_prompt
            
            # 文本生成
            outputs = self.generate(
                input_ids=current_input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **generate_kwargs
            )
            
            generated_ids = outputs.sequences
            
            # ========== Stage 4: Mask Generation ==========
            pred_masks = None
            if self.model.seg_token_idx in generated_ids:
                # 使用 forward 方法生成掩码
                forward_outputs = self.forward(
                    input_ids=generated_ids,
                    images=images,
                    output_hidden_states=True
                )
                pred_masks = forward_outputs['pred_masks']
                hidden_states = forward_outputs['hidden_states']
                
                # ========== Stage 4: Consistency Check ==========
                if pred_masks is not None and enable_self_correction:
                    consistency_outputs = self.model.consistency_checker(
                        mask_output=pred_masks,
                        text_embeds=hidden_states,
                        return_attention=False
                    )
                    consistency_score = consistency_outputs['consistency_score'].item()
                    
                    # 记录当前结果
                    current_result = {
                        'generated_ids': generated_ids,
                        'pred_masks': pred_masks,
                        'consistency_score': consistency_score,
                        'iteration': iteration
                    }
                    correction_history.append(current_result)
                    
                    # 更新最佳结果
                    if consistency_score > best_score:
                        best_score = consistency_score
                        best_result = current_result
                    
                    # 检查是否达到阈值
                    if consistency_score >= consistency_threshold:
                        break
                else:
                    # 如果没有启用自我修正，直接返回
                    best_result = {
                        'generated_ids': generated_ids,
                        'pred_masks': pred_masks,
                        'consistency_score': 1.0,
                        'iteration': 0
                    }
                    break
            else:
                # 如果没有生成掩码，返回文本生成结果
                best_result = {
                    'generated_ids': generated_ids,
                    'pred_masks': None,
                    'consistency_score': 1.0,
                    'iteration': 0
                }
                break
            
            # 如果是最后一次迭代，退出循环
            if not enable_self_correction or iteration == max_iterations - 1:
                break
        
        # ========== Return Best Result ==========
        correction_info = {
            'num_iterations': len(correction_history),
            'final_score': best_score if best_result else 0.0,
            'history': correction_history,
            'improved': len(correction_history) > 1 and best_score > correction_history[0]['consistency_score']
        }
        
        if best_result is None:
            return generated_ids, None, correction_info
        
        return best_result['generated_ids'], best_result['pred_masks'], correction_info
    
    def _build_refinement_prompt(
        self,
        previous_ids: torch.Tensor,
        consistency_score: float
    ) -> torch.Tensor:
        """
        构建修正提示
        
        将之前的生成结果作为负面反馈，引导模型改进
        """
        # 简化版：在原始输入后添加修正指令
        # 实际应用中可以使用 tokenizer 构建更复杂的提示
        # 这里假设调用方会处理 tokenization
        
        # TODO: 实现更复杂的提示工程
        # 例如："The previous report (score={score:.2f}) was inconsistent. Please refine."
        
        return previous_ids  # 简化版：直接返回（需要外部处理）


# ========== Backward Compatibility ==========
# 保留原有类名以兼容现有代码
Med3DLISA = Med3DLISA_Full


# ========== Utility Functions ==========
def create_med3d_lisa_model(
    config: Med3DLISAConfig,
    pretrained_llm_path: Optional[str] = None,
    load_full_model: bool = True
) -> Med3DLISA_Full:
    """
    创建 Med3D-LISA 模型的辅助函数
    
    Args:
        config: 模型配置
        pretrained_llm_path: 预训练 LLM 路径（可选）
        load_full_model: 是否加载完整模型（包括新增模块）
    
    Returns:
        Med3DLISA_Full 实例
    """
    model = Med3DLISA_Full(config)
    
    if pretrained_llm_path is not None:
        # 加载预训练 LLM 权重
        # TODO: 实现选择性权重加载
        pass
    
    return model