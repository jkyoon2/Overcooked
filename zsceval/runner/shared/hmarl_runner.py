import copy
import numpy as np
import torch
from zsceval.runner.separated.base_runner import _t2n
from zsceval.runner.separated.hmarl_runner import HMARLRunner as SeparatedHMARLRunner
from zsceval.utils.hmarl_buffer import LowLevelRolloutBuffer

class HMARLRunner(SeparatedHMARLRunner):
    """
    Shared Policy Runner:
    부모 클래스(Separated)의 Optuna 로직(run 메소드)은 그대로 물려받고,
    데이터를 하나로 합치는(Flatten) 로직만 재정의합니다.
    """
    def __init__(self, config):
        super().__init__(config) # 부모의 init (Optuna 설정 등 포함) 호출

        # [Shared 전용] 배치 사이즈를 (Thread * Agent)로 확장
        low_args = copy.copy(self.all_args)
        low_args.n_rollout_threads = self.n_rollout_threads * self.num_agents

        share_obs_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )
        
        # [Shared 전용] 단일 버퍼 생성
        self.low_buffer = LowLevelRolloutBuffer(
            low_args,
            self.envs.observation_space[0],
            share_obs_space,
            self.envs.action_space[0],
            self.skill_dim,
        )
        
        # 부모 클래스와의 호환성을 위해 리스트에도 넣어둠 (하지만 실제로는 self.low_buffer만 씀)
        self.low_buffers = [self.low_buffer]
        
        # [Shared 전용] 단일 트레이너/폴리시 연결
        self.low_trainer = self.trainer.low_trainers[0]
        self.low_trainers = [self.low_trainer]
        self.low_policy = self.policy.low_levels[0]

    # --- Helper Methods for Flattening ---
    def _flatten_env_agent(self, arr):
        # (50, 2, ...) -> (100, ...)
        return arr.reshape(self.n_rollout_threads * self.num_agents, *arr.shape[2:])

    def _unflatten_env_agent(self, arr):
        # (100, ...) -> (50, 2, ...)
        return arr.reshape(self.n_rollout_threads, self.num_agents, *arr.shape[1:])

    # --- Overridden Methods (데이터 처리 방식 변경) ---

    def warmup(self):
        # 환경 초기화
        obs_batch, info_list = self.envs.reset()
        all_agent_obs = np.array([info["all_agent_obs"] for info in info_list])
        share_obs = np.array([info["share_obs"] for info in info_list])
        available_actions = np.array([info["available_actions"] for info in info_list])

        self.segment_start_obs = all_agent_obs
        self.segment_start_share_obs = share_obs
        
        # 스킬 샘플링
        self.current_skills, self.rnn_states_actor_high = self.sample_skills(
            share_obs, all_agent_obs, 0, None, masks=self.high_masks
        )
        
        if self.use_constraint:
            self._update_constraint_penalty()

        # [Shared 전용] 데이터를 Flatten해서 단일 버퍼에 넣음
        buffer_share_obs = share_obs if self.use_centralized_V else all_agent_obs
        self.low_buffer.share_obs[0] = self._flatten_env_agent(buffer_share_obs)
        self.low_buffer.obs[0] = self._flatten_env_agent(all_agent_obs)
        self.low_buffer.skills[0] = self.current_skills.reshape(-1, self.skill_dim)
        if self.low_buffer.available_actions is not None:
            self.low_buffer.available_actions[0] = self._flatten_env_agent(available_actions)

    @torch.no_grad()
    def collect(self, step):
        self.low_policy.prep_rollout()

        # [Shared 전용] RNN State Flatten (50, 2, 1, 64) -> (100, 1, 64)
        rnn_states = self.rnn_states_low.transpose(1, 0, 2, 3).reshape(
            -1, self.rnn_states_low.shape[2], self.rnn_states_low.shape[3]
        )
        rnn_states_critic = self.rnn_states_low_critic.transpose(1, 0, 2, 3).reshape(
            -1, self.rnn_states_low_critic.shape[2], self.rnn_states_low_critic.shape[3]
        )
        skills_flat = self.current_skills.reshape(-1, self.skill_dim)

        # 단일 Policy로 행동 결정
        value, action, action_log_prob, next_rnn, next_rnn_critic = self.low_policy.get_actions(
            self.low_buffer.share_obs[step],
            self.low_buffer.obs[step],
            rnn_states,
            rnn_states_critic,
            self.low_buffer.masks[step],
            self.low_buffer.available_actions[step] if self.low_buffer.available_actions is not None else None,
            deterministic=False,
            skill=skills_flat,
        )

        # [Shared 전용] 부모 run()이 이해할 수 있도록 다시 (50, 2, ...) 형태로 복구해서 리턴
        values_env = self._unflatten_env_agent(_t2n(value))
        actions_env = self._unflatten_env_agent(_t2n(action))
        action_log_probs_env = self._unflatten_env_agent(_t2n(action_log_prob))
        rnn_states_env = self._unflatten_env_agent(_t2n(next_rnn))
        rnn_states_critic_env = self._unflatten_env_agent(_t2n(next_rnn_critic))
        
        return values_env, actions_env, action_log_probs_env, rnn_states_env, rnn_states_critic_env

    def insert(self, data):
        # 부모 run()이 넘겨준 (50, 2, ...) 형태의 데이터를 받음
        (
            obs, share_obs, rewards, dones, infos, available_actions,
            values, actions, action_log_probs, rnn_states, rnn_states_critic,
        ) = data

        masks = 1.0 - dones.astype(np.float32)[..., None]
        masks_flat = masks.reshape(-1, 1)

        buffer_share_obs = share_obs if self.use_centralized_V else obs
        
        # [Shared 전용] 전부 Flatten 처리
        share_obs_flat = self._flatten_env_agent(buffer_share_obs)
        obs_flat = self._flatten_env_agent(obs)
        skills_flat = self.current_skills.reshape(-1, self.skill_dim)
        values_flat = self._flatten_env_agent(values)
        actions_flat = self._flatten_env_agent(actions)
        action_log_probs_flat = self._flatten_env_agent(action_log_probs)
        rewards_flat = rewards.reshape(-1, 1) # Reward는 (100, 1)
        rnn_states_flat = self._flatten_env_agent(rnn_states)
        rnn_states_critic_flat = self._flatten_env_agent(rnn_states_critic)
        
        if available_actions is not None:
            available_actions_flat = self._flatten_env_agent(available_actions)
        else:
            available_actions_flat = None

        # 단일 버퍼에 삽입
        self.low_buffer.insert(
            share_obs_flat, obs_flat, skills_flat, rnn_states_flat, rnn_states_critic_flat,
            actions_flat, action_log_probs_flat, values_flat, rewards_flat, masks_flat,
            available_actions=available_actions_flat,
        )

        # 다음 스텝을 위해 RNN State 저장 (다시 Agent 차원 분리)
        self.rnn_states_low = rnn_states.transpose(1, 0, 2, 3)
        self.rnn_states_low_critic = rnn_states_critic.transpose(1, 0, 2, 3)

    def finish_segment(self, obs, share_obs, dones, ep_intrinsic_r, ep_low_total_r, num_steps):
        self.low_trainer.adapt_intrinsic_temp(num_steps)
        self.high_trainer.adapt_gumbel_tau(num_steps)

        # High Level Buffer 처리는 부모와 유사하지만 Flatten/Unflatten 주의
        next_masks = 1.0 - dones.astype(np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)
        obs_channels, grid_shape = self._get_obs_channels_and_grid()

        adjusted_reward = self.segment_reward
        if self.use_constraint:
            adjusted_reward = self.segment_reward - self.segment_penalty

        # High Buffer는 (Batch, ...) 형태 유지 (Flatten 안 함)
        for env_idx in range(self.n_rollout_threads):
            self.high_buffer.add(
                self.segment_start_obs[env_idx],
                self.segment_start_share_obs[env_idx],
                self.current_skills[env_idx],
                adjusted_reward[env_idx].copy() / 10.0,
                obs[env_idx],
                share_obs[env_idx],
                dones[env_idx].astype(np.float32).reshape(-1, 1),
                mask=self.prev_masks_high[env_idx],
                next_mask=next_masks[env_idx],
                rnn_states_actor=self.prev_rnn_states_actor_high[env_idx],
                rnn_states_critic=self.prev_rnn_states_critic_high[env_idx],
                next_rnn_states_actor=self.rnn_states_actor_high[env_idx],
                next_rnn_states_critic=self.rnn_states_critic_high[env_idx],
            )

        # Intrinsic Reward 계산 및 Buffer 업데이트 (여기가 가장 중요)
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                seg_len = len(self.segment_traj_obs[e][a])
                if seg_len == 0:
                    continue

                # VAE 입력 포맷팅
                obs_seq_arr = self._format_obs_seq(np.stack(self.segment_traj_obs[e][a]), obs_channels, grid_shape)
                if obs_seq_arr is None:
                    raise ValueError("Failed to format obs sequence.")
                obs_seq = torch.as_tensor(obs_seq_arr[None], device=self.device).float()
                act_seq = torch.as_tensor(np.stack(self.segment_traj_actions[e][a])[None], device=self.device)
                skill = torch.as_tensor(self.current_skills[e, a][None], device=self.device)

                # Intrinsic 계산
                intrinsic_val = self.low_trainer.compute_intrinsic_reward_value(obs_seq, act_seq, skill)
                ep_intrinsic_r[e, a] += intrinsic_val

                per_step_intrinsic = intrinsic_val / max(1, seg_len)

                # 버퍼 내의 Reward 업데이트 (Flatten된 인덱스 계산 필요)
                start_idx = (self.low_buffer.step - seg_len) % self.episode_length
                idxs = [(start_idx + k) % self.episode_length for k in range(seg_len)]
                
                # [Shared 전용] Flat Index 계산: env_idx * num_agents + agent_idx
                flat_idx = e * self.num_agents + a
                
                current_extrinsic = self.low_buffer.rewards[idxs, flat_idx, 0] / 10.0
                alpha = self.low_trainer.intrinsic_alpha
                new_reward = alpha * current_extrinsic + (1 - alpha) * per_step_intrinsic
                self.low_buffer.rewards[idxs, flat_idx, 0] = new_reward

                ep_low_total_r[e, a] += float(np.sum(new_reward))

        # 리셋 로직
        self.segment_reward[:] = 0
        self.segment_step[:] = 0
        self.segment_start_obs = obs
        self.segment_start_share_obs = share_obs
        self.high_masks = next_masks
        self.segment_traj_obs = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        self.segment_traj_actions = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        self.current_skills, self.rnn_states_actor_high = self.sample_skills(
            share_obs, obs, 0, None, masks=self.high_masks
        )
        if self.use_constraint:
            self._update_constraint_penalty()

    @torch.no_grad()
    def compute(self):
        # [Shared 전용] 단일 정책으로 Value 계산
        next_value = self.low_policy.get_values(
            self.low_buffer.share_obs[-1],
            self.low_buffer.rnn_states_critic[-1],
            self.low_buffer.masks[-1],
            skill=self.low_buffer.skills[-1],
        )
        next_value = _t2n(next_value)
        self.low_buffer.compute_returns(next_value, self.low_trainer.value_normalizer)

    def train(self, num_steps):
        # [Shared 전용] 단일 트레이너로 1회 업데이트
        self.low_trainer.prep_training()
        self.low_trainer.adapt_entropy_coef(num_steps)
        train_info = self.low_trainer.train(self.low_buffer)
        self.low_buffer.after_update()

        high_info = None
        if len(self.high_buffer) >= self.high_batch_size:
            self.high_trainer.prep_training()
            high_info = self.high_trainer.train(self.high_buffer, batch_size=self.high_batch_size)

        # 부모 run()은 'low' 키에 리스트 형태의 정보를 기대하므로 형식 맞춰줌
        return {"low": [dict(train_info) for _ in range(self.num_agents)], "high": high_info}

    def save(self, steps=None):
        if not getattr(self.all_args, "share_policy", False):
            super().save(steps)
            return

        postfix = f"_{steps}.pt" if steps else ".pt"
        # Shared Policy는 하나만 저장하면 됨
        torch.save(self.low_policies[0].actor.state_dict(), str(self.save_dir / f"ll_actor{postfix}"))
        torch.save(self.low_policies[0].critic.state_dict(), str(self.save_dir / f"ll_critic{postfix}"))
        torch.save(self.high_policy.critic.state_dict(), str(self.save_dir / f"hl_critic{postfix}"))
        torch.save(self.high_policy.actors[0].state_dict(), str(self.save_dir / f"hl_actor{postfix}"))
