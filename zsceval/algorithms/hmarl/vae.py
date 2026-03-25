import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    """
    Simple CNN with LayerNorm to prevent feature collapse
    Input: (B, C, H, W) -> Output: (B, feature_dim)
    """
    def __init__(self, in_channels, feature_dim, input_hw=None):
        super().__init__()
        if input_hw is None:
            input_hw = (5, 13)
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        linear_in = 64 * self.input_hw[0] * self.input_hw[1]
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_in, feature_dim),
            nn.LayerNorm(feature_dim),           # [핵심] 0으로 죽는 것 방지
            nn.ReLU()
        )

    def _pad_to_input_hw(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        target_h, target_w = self.input_hw
        if h > target_h or w > target_w:
            raise ValueError(
                f"GridEncoder input {h}x{w} exceeds configured input_hw {self.input_hw}."
            )
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, x):
        if x.shape[-2:] != self.input_hw:
            x = self._pad_to_input_hw(x)
        return self.net(x)


class TrajectoryEncoder(nn.Module):
    def __init__(self, obs_channels, action_dim, task_dim, hidden_dim, latent_dim, input_hw=None):
        super().__init__()
        self.feature_dim = 256
        
        # 1. Grid Feature Extractor
        self.cnn = GridEncoder(obs_channels, self.feature_dim, input_hw=input_hw)
        
        # 2. Embeddings for input balancing
        self.action_emb = nn.Linear(action_dim, 32)
        self.task_emb = nn.Linear(task_dim, 32)
        
        # 3. Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=self.feature_dim + 32 + 32,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc_z = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, obs_seq, action_seq, task_id, return_features=False):
        B, T, C, H, W = obs_seq.shape
        
        # CNN (Time-distributed)
        obs_reshaped = obs_seq.contiguous().reshape(B * T, C, H, W)
        cnn_features = self.cnn(obs_reshaped).reshape(B, T, self.feature_dim)
        
        # Task ID: repeat to match B if needed
        if task_id.shape[0] != B:
            repeat_factor = B // task_id.shape[0]
            task_id = task_id.repeat(repeat_factor, 1)
        
        # Embeddings
        act_emb = self.action_emb(action_seq)  # (B, T, 32)
        task_emb = self.task_emb(task_id)  # (B, 32)
        task_seq_emb = task_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 32)
        
        # Concat & LSTM
        lstm_input = torch.cat([cnn_features, act_emb, task_seq_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Mean Pooling
        pooled = lstm_out.mean(dim=1)
        z_e = self.fc_z(pooled)
        
        if return_features:
            return z_e, cnn_features
        return z_e


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module with straight-through estimator.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_beta: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, D)
        flat_z = z_e.view(-1, self.embedding_dim)

        # Normalize for cosine-similarity-based mapping.
        flat_z = F.normalize(flat_z, p=2, dim=1)
        z_e_norm = flat_z.view_as(z_e)

        with torch.no_grad():
            self.embedding.weight.copy_(F.normalize(self.embedding.weight, p=2, dim=1))

        # Compute L2 distance to embeddings
        emb_weights = self.embedding.weight
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(emb_weights ** 2, dim=1)
            - 2 * torch.matmul(flat_z, emb_weights.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(encoding_indices, emb_weights).view_as(z_e)

        codebook_loss = F.mse_loss(z_q, z_e_norm.detach())
        commitment_loss = F.mse_loss(z_e_norm, z_q.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss

        # Straight-through estimator
        z_q_st = z_e_norm + (z_q - z_e_norm).detach()

        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, codebook_loss, commitment_loss, perplexity, encoding_indices


class TrajectoryDecoder(nn.Module):
    """
    MLP Decoder: z -> All Timesteps Features & Actions
    """
    def __init__(self, latent_dim, task_dim, hidden_dim, feature_dim, action_dim, t_seg):
        super().__init__()
        self.t_seg = t_seg
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        decoder_hidden = hidden_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + task_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU()
        )
        
        self.feature_head = nn.Linear(decoder_hidden, t_seg * feature_dim)
        self.action_head = nn.Linear(decoder_hidden, t_seg * action_dim)
    
    def forward(self, z, task_id):
        z_task = torch.cat([z, task_id], dim=-1)
        common_feat = self.net(z_task)
        
        pred_features = self.feature_head(common_feat).view(-1, self.t_seg, self.feature_dim)
        pred_actions = self.action_head(common_feat).view(-1, self.t_seg, self.action_dim)
        
        return pred_features, pred_actions


class SkillVAE(nn.Module):
    def __init__(
        self,
        obs_channels,
        action_dim,
        task_dim,
        hidden_dim=128,
        latent_dim=16,
        t_seg=5,
        codebook_size: int = 128,
        commitment_beta: float = 0.25,
        grid_shape=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.t_seg = t_seg
        self.grid_shape = grid_shape
        self.encoder = TrajectoryEncoder(
            obs_channels, action_dim, task_dim, hidden_dim, latent_dim, input_hw=grid_shape
        )
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment_beta)
        self.decoder = TrajectoryDecoder(latent_dim, task_dim, hidden_dim, 
                                       feature_dim=self.encoder.feature_dim,
                                       action_dim=action_dim,
                                       t_seg=t_seg)
    
    def forward(self, obs_seq, action_seq, task_id):
        z_e, target_features = self.encoder(obs_seq, action_seq, task_id, return_features=True)
        z_q, vq_loss, codebook_loss, commitment_loss, perplexity, encoding_indices = self.quantizer(z_e)
        pred_features, pred_actions = self.decoder(z_q, task_id)
        return (
            z_e,
            z_q,
            pred_features,
            pred_actions,
            target_features,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
            encoding_indices,
        )
    
    def encode(self, obs_seq, action_seq, task_id, deterministic: bool = False, return_indices: bool = False):
        """
        Encode trajectory to latent skill
        
        Args:
            obs_seq: (B, T, C, H, W) observation sequence
            action_seq: (B, T, action_dim) one-hot actions
            task_id: (B, task_dim) one-hot task identifier
            deterministic: unused for VQ-VAE (kept for API compatibility)
            return_indices: if True, also return codebook indices
        
        Returns:
            z_q: (B, latent_dim) quantized skill embedding
        """
        z_e = self.encoder(obs_seq, action_seq, task_id)
        z_q, _, _, _, _, indices = self.quantizer(z_e)
        if return_indices:
            return z_q, indices
        return z_q
    
    def decode(self, z, task_id):
        """
        Decode latent skill to reconstructed features and actions
        
        Args:
            z: (B, latent_dim) latent skill
            task_id: (B, task_dim) one-hot task identifier
        
        Returns:
            pred_features: (B, T, feature_dim)
            pred_actions: (B, T, action_dim)
        """
        return self.decoder(z, task_id)

    def compute_loss(
        self,
        obs_seq,
        action_seq,
        task_id,
        feature_weight: float = 300.0,
        action_weight: float = 10.0,
    ):
        """
        Feature and action reconstruction loss with scaling
        
        Args:
            feature_weight: Weight for feature reconstruction loss
            action_weight: Weight for action reconstruction loss
        """
        (
            z_e,
            z_q,
            pred_features,
            pred_actions,
            target_features,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
            _,
        ) = self.forward(obs_seq, action_seq, task_id)
        
        recon_feat_loss = F.mse_loss(pred_features, target_features)

        _, _, A_dim = pred_actions.shape

        # Convert action_seq to target indices -> one-hot to indices (class integer for CrossEntropyLoss)    
        target_action_indices = action_seq.argmax(dim=-1).view(-1)

        # Flatten predictions and targets for CrossEntropyLoss
        pred_actions = pred_actions.view(-1, A_dim)

        # Cross Entropy Loss (Softmax 내부적 적용)
        recon_act_loss = F.cross_entropy(pred_actions, target_action_indices)
        
        scaled_feat_loss = recon_feat_loss * feature_weight
        scaled_act_loss = recon_act_loss * action_weight
        recon_loss = scaled_feat_loss + scaled_act_loss
        
        total_loss = recon_loss + vq_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'recon': recon_loss.item(),
            'recon_f': scaled_feat_loss.item(),
            'recon_a': scaled_act_loss.item(),
            'vq_loss': vq_loss.item(),
            'codebook': codebook_loss.item(),
            'commitment': commitment_loss.item(),
            'perplexity': perplexity.item()
        }
