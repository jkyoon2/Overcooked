import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class GridEncoder(nn.Module):
    """
    (변경 없음) Simple CNN with LayerNorm
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
            nn.LayerNorm(feature_dim),
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
    # [변경] skill_range 파라미터 추가
    def __init__(self, obs_channels, action_dim, task_dim, hidden_dim, latent_dim, skill_range=2.0, input_hw=None):
        super().__init__()
        self.feature_dim = 256
        self.skill_range = skill_range  # [핵심] 스킬 범위 저장
        
        # 1. Grid Feature Extractor
        self.cnn = GridEncoder(obs_channels, self.feature_dim, input_hw=input_hw)
        
        # 2. Embeddings
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
        
        # [변경] mu, logvar 대신 그냥 z 하나만 예측
        self.fc_z = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, obs_seq, action_seq, task_id, return_features=False):
        B, T, C, H, W = obs_seq.shape
        
        # ... (이전과 동일: CNN, Embedding, LSTM) ...
        obs_reshaped = obs_seq.contiguous().reshape(B * T, C, H, W)
        cnn_features = self.cnn(obs_reshaped).reshape(B, T, self.feature_dim)
        
        if task_id.shape[0] != B:
            repeat_factor = B // task_id.shape[0]
            task_id = task_id.repeat(repeat_factor, 1)
            
        act_emb = self.action_emb(action_seq)
        task_emb = self.task_emb(task_id)
        task_seq_emb = task_emb.unsqueeze(1).expand(-1, T, -1)
        
        lstm_input = torch.cat([cnn_features, act_emb, task_seq_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Mean Pooling
        pooled = lstm_out.mean(dim=1)
        
        # [핵심 변경] Deterministic Encoding with Tanh Bounding
        raw_z = self.fc_z(pooled)
        z = torch.tanh(raw_z) * self.skill_range  # 무조건 [-range, +range] 안에 들어감
        
        if return_features:
            return z, cnn_features  # mu, logvar 삭제됨
        return z

    # sample 함수 삭제됨 (필요 없음)


class TrajectoryDecoder(nn.Module):
    """ (변경 없음) MLP Decoder """
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


class SkillAE(nn.Module):  # 이름 변경: VAE -> AE
    def __init__(self, obs_channels, action_dim, task_dim, hidden_dim=128, latent_dim=16, t_seg=5, skill_range=2.0, grid_shape=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.t_seg = t_seg
        # [변경] skill_range 전달
        self.encoder = TrajectoryEncoder(
            obs_channels, action_dim, task_dim, hidden_dim, latent_dim, skill_range, input_hw=grid_shape
        )
        self.decoder = TrajectoryDecoder(latent_dim, task_dim, hidden_dim, 
                                       feature_dim=self.encoder.feature_dim,
                                       action_dim=action_dim,
                                       t_seg=t_seg)
    
    def forward(self, obs_seq, action_seq, task_id):
        # [변경] 리턴값 단순화 (mu, logvar 없음)
        z, target_features = self.encoder(obs_seq, action_seq, task_id, return_features=True)
        pred_features, pred_actions = self.decoder(z, task_id)
        return z, pred_features, pred_actions, target_features
    
    def encode(self, obs_seq, action_seq, task_id):
        """ deterministic 파라미터 삭제 (항상 deterministic) """
        z = self.encoder(obs_seq, action_seq, task_id)
        return z
    
    def decode(self, z, task_id):
        return self.decoder(z, task_id)

    # [변경] beta 파라미터 삭제, KL Loss 삭제
    def compute_loss(self, obs_seq, action_seq, task_id, 
                     feature_weight: float = 2000.0, action_weight: float = 5.0):
        
        z, pred_features, pred_actions, target_features = self.forward(obs_seq, action_seq, task_id)
        
        recon_feat_loss = F.mse_loss(pred_features, target_features)
        recon_act_loss = F.mse_loss(pred_actions, action_seq)
        
        scaled_feat_loss = recon_feat_loss * feature_weight
        scaled_act_loss = recon_act_loss * action_weight
        
        # [핵심] KL Loss 없이 오직 복원 손실만!
        total_loss = scaled_feat_loss + scaled_act_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'recon_f': scaled_feat_loss.item(),
            'recon_a': scaled_act_loss.item(),
            # KL은 이제 0입니다 (기록용으로 남겨둘 수도 있음)
            'kl': 0.0,
            'z_mean': z.mean().item(), # z가 잘 퍼져있는지 확인용
            'z_std': z.std().item()
        }
