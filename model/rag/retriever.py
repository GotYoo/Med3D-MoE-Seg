"""
Medical Knowledge Retriever
Stage 2: 医学知识库检索与上下文注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import os
import numpy as np


class MedicalKnowledgeRetriever(nn.Module):
    """
    医学知识检索器
    
    功能：
    1. 从医学知识库中检索相关信息（模拟 FAISS 索引）
    2. 基于 Cosine Similarity 检索 Top-K 相关知识
    3. 将检索到的知识投影并准备注入到 LLM
    """
    
    def __init__(self, 
                 knowledge_embed_path: Optional[str] = None,
                 knowledge_texts_path: Optional[str] = None,
                 knowledge_dim: int = 768,
                 llm_hidden_size: int = 4096,
                 top_k: int = 3,
                 use_learnable_kb: bool = False,
                 num_knowledge_entries: int = 1000):
        """
        初始化医学知识检索器
        
        Args:
            knowledge_embed_path: 知识库 Embedding 矩阵路径（如果为 None 则使用随机初始化）
            knowledge_texts_path: 知识库文本路径（JSON 格式）
            knowledge_dim: 知识库 Embedding 维度
            llm_hidden_size: LLM 的 hidden size，用于投影
            top_k: 检索返回的 Top-K 数量
            use_learnable_kb: 是否使用可学习的知识库（用于训练）
            num_knowledge_entries: 知识库条目数量（如果使用随机初始化）
        """
        super().__init__()
        
        self.knowledge_dim = knowledge_dim
        self.llm_hidden_size = llm_hidden_size
        self.top_k = top_k
        self.use_learnable_kb = use_learnable_kb
        self.knowledge_texts = None
        
        # 加载或初始化知识库 Embedding 矩阵
        if knowledge_embed_path is not None and os.path.exists(knowledge_embed_path):
            # 从文件加载知识库
            print(f"Loading knowledge base from {knowledge_embed_path}")
            knowledge_embeds = torch.load(knowledge_embed_path)
            self.num_knowledge_entries = knowledge_embeds.shape[0]
            
            # 加载知识文本
            if knowledge_texts_path is not None and os.path.exists(knowledge_texts_path):
                import json
                with open(knowledge_texts_path, 'r', encoding='utf-8') as f:
                    self.knowledge_texts = json.load(f)
                print(f"Loaded {len(self.knowledge_texts)} knowledge texts from {knowledge_texts_path}")
        else:
            # 使用随机初始化（模拟知识库）
            print(f"Initializing random knowledge base with {num_knowledge_entries} entries")
            self.num_knowledge_entries = num_knowledge_entries
            knowledge_embeds = torch.randn(num_knowledge_entries, knowledge_dim)
            # 归一化
            knowledge_embeds = F.normalize(knowledge_embeds, dim=-1)
        
        # 注册为 buffer 或 parameter
        if use_learnable_kb:
            # 可学习的知识库（用于端到端训练）
            self.knowledge_embeds = nn.Parameter(knowledge_embeds)
        else:
            # 固定的知识库（不参与梯度更新）
            self.register_buffer('knowledge_embeds', knowledge_embeds)
        
        # 上下文投影层：将检索到的知识投影到 LLM hidden size
        self.context_projector = nn.Sequential(
            nn.Linear(knowledge_dim * top_k, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
        # 查询投影层（可选）：确保 query 与知识库在同一空间
        self.query_projector = nn.Linear(knowledge_dim, knowledge_dim)
        
        print(f"MedicalKnowledgeRetriever initialized:")
        print(f"  - Knowledge entries: {self.num_knowledge_entries}")
        print(f"  - Knowledge dim: {knowledge_dim}")
        print(f"  - Top-K: {top_k}")
        print(f"  - LLM hidden size: {llm_hidden_size}")
    
    def retrieve(self, 
                 query_embed: torch.Tensor,
                 top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检索相关知识
        
        Args:
            query_embed: 查询 Embedding [B, knowledge_dim] 或 [B, seq_len, knowledge_dim]
            top_k: 检索 Top-K（如果为 None 则使用初始化时的值）
            
        Returns:
            retrieved_embeds: 检索到的知识 Embeddings [B, top_k, knowledge_dim]
            retrieved_indices: 检索到的知识索引 [B, top_k]
            relevance_scores: 相关性分数 [B, top_k]
        """
        if top_k is None:
            top_k = self.top_k
        
        # 处理输入维度
        if query_embed.dim() == 3:
            # [B, seq_len, D] -> 取平均或最后一个 token
            query_embed = query_embed.mean(dim=1)  # [B, D]
        
        # 投影查询（可选）
        query_embed = self.query_projector(query_embed)  # [B, D]
        
        # L2 归一化（用于 Cosine Similarity）
        query_embed = F.normalize(query_embed, dim=-1)  # [B, D]
        
        # [Device Fix] Ensure knowledge_embeds is on correct device
        kb_data = self.knowledge_embeds
        if kb_data.device != query_embed.device:
            kb_data = kb_data.to(query_embed.device)
            
        knowledge_embeds = F.normalize(kb_data, dim=-1)  # [N, D]
        
        # 计算 Cosine Similarity：[B, N]
        similarity = torch.matmul(query_embed, knowledge_embeds.t())  # [B, N]
        
        # 检索 Top-K
        relevance_scores, retrieved_indices = torch.topk(
            similarity, k=top_k, dim=-1, largest=True, sorted=True
        )  # [B, top_k], [B, top_k]
        
        # 获取检索到的知识 Embeddings
        batch_size = query_embed.shape[0]
        retrieved_embeds = []
        for i in range(batch_size):
            retrieved_embeds.append(kb_data[retrieved_indices[i]])
        retrieved_embeds = torch.stack(retrieved_embeds, dim=0)  # [B, top_k, D]
        
        return retrieved_embeds, retrieved_indices, relevance_scores
    
    def forward(self, 
                query_embed: torch.Tensor,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播：检索 + 投影
        
        Args:
            query_embed: 查询 Embedding [B, knowledge_dim]
            return_details: 是否返回检索细节
            
        Returns:
            outputs: 包含投影后的上下文的字典
                - context_embed: 投影后的上下文 [B, llm_hidden_size]
                - retrieved_embeds: 检索到的原始 Embeddings [B, top_k, knowledge_dim] (可选)
                - retrieved_indices: 检索到的索引 [B, top_k] (可选)
                - relevance_scores: 相关性分数 [B, top_k] (可选)
                - retrieved_texts: 检索到的知识文本 List[List[Dict]] (可选)
        """
        # 执行检索
        retrieved_embeds, retrieved_indices, relevance_scores = self.retrieve(query_embed)
        # retrieved_embeds: [B, top_k, knowledge_dim]
        
        # 展平并投影到 LLM hidden size
        batch_size = retrieved_embeds.shape[0]
        retrieved_flat = retrieved_embeds.view(batch_size, -1)  # [B, top_k * knowledge_dim]
        context_embed = self.context_projector(retrieved_flat)  # [B, llm_hidden_size]
        
        # 构建输出
        outputs = {
            'context_embed': context_embed,
        }
        
        if return_details:
            outputs.update({
                'retrieved_embeds': retrieved_embeds,
                'retrieved_indices': retrieved_indices,
                'relevance_scores': relevance_scores,
            })
            
            # 添加检索到的文本（如果可用）
            if self.knowledge_texts is not None:
                retrieved_texts = []
                for batch_idx in range(batch_size):
                    batch_texts = []
                    for k_idx in retrieved_indices[batch_idx].cpu().tolist():
                        if 0 <= k_idx < len(self.knowledge_texts):
                            batch_texts.append(self.knowledge_texts[k_idx])
                    retrieved_texts.append(batch_texts)
                outputs['retrieved_texts'] = retrieved_texts
        
        return outputs
    
    def inject_context_to_prompt(self,
                                  prompt_embeds: torch.Tensor,
                                  context_embed: torch.Tensor,
                                  injection_position: str = 'prepend') -> torch.Tensor:
        """
        将检索到的上下文注入到 Prompt Embeddings
        
        Args:
            prompt_embeds: 原始 Prompt Embeddings [B, seq_len, llm_hidden_size]
            context_embed: 上下文 Embedding [B, llm_hidden_size]
            injection_position: 注入位置 ('prepend', 'append', 'middle')
            
        Returns:
            enhanced_embeds: 注入上下文后的 Embeddings [B, seq_len+1, llm_hidden_size]
        """
        batch_size = prompt_embeds.shape[0]
        
        # 扩展 context_embed 维度
        context_embed = context_embed.unsqueeze(1)  # [B, 1, llm_hidden_size]
        
        if injection_position == 'prepend':
            # 在开头注入
            enhanced_embeds = torch.cat([context_embed, prompt_embeds], dim=1)
        elif injection_position == 'append':
            # 在末尾注入
            enhanced_embeds = torch.cat([prompt_embeds, context_embed], dim=1)
        elif injection_position == 'middle':
            # 在中间注入
            mid_pos = prompt_embeds.shape[1] // 2
            enhanced_embeds = torch.cat([
                prompt_embeds[:, :mid_pos, :],
                context_embed,
                prompt_embeds[:, mid_pos:, :]
            ], dim=1)
        else:
            raise ValueError(f"Unknown injection_position: {injection_position}")
        
        return enhanced_embeds
    
    def save_knowledge_base(self, path: str):
        """保存知识库到文件"""
        torch.save(self.knowledge_embeds.detach().cpu(), path)
        print(f"Knowledge base saved to {path}")
    
    def load_knowledge_base(self, path: str):
        """从文件加载知识库"""
        knowledge_embeds = torch.load(path, map_location=self.knowledge_embeds.device)
        if self.use_learnable_kb:
            self.knowledge_embeds = nn.Parameter(knowledge_embeds)
        else:
            self.register_buffer('knowledge_embeds', knowledge_embeds)
        print(f"Knowledge base loaded from {path}")


def create_dummy_knowledge_base(num_entries: int = 1000, 
                                embed_dim: int = 768,
                                save_path: Optional[str] = None) -> torch.Tensor:
    """
    创建模拟的医学知识库
    
    Args:
        num_entries: 知识条目数量
        embed_dim: Embedding 维度
        save_path: 保存路径（可选）
        
    Returns:
        knowledge_embeds: 知识库 Embeddings [num_entries, embed_dim]
    """
    # 生成随机 Embeddings（在实际应用中应该是真实的医学知识编码）
    knowledge_embeds = torch.randn(num_entries, embed_dim)
    knowledge_embeds = F.normalize(knowledge_embeds, dim=-1)
    
    if save_path is not None:
        torch.save(knowledge_embeds, save_path)
        print(f"Dummy knowledge base saved to {save_path}")
    
    return knowledge_embeds
