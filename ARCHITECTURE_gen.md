# Robomimic Consistency Policy Architecture

This note summarizes the Robomimic image-based Consistency Policy used in this
repo, based on the loaded checkpoints:

- `kdh/checkpoint/consistency_lift_best.ckpt`
- `kdh/checkpoint/consistency_can_best.ckpt`
- `kdh/checkpoint/consistency_square_best.ckpt`

The policy class is:

```text
consistency_policy.student.ctm_policy.CTMPPUnetHybridImagePolicy
```

## Data

Training uses Robomimic expert demonstration data from `image_abs.hdf5`.
Each sample contains a short observation history and an expert action
trajectory.

For lift, can, and square, the observation schema is:

```text
agentview_image              (B, 2, 3, 84, 84)
robot0_eye_in_hand_image     (B, 2, 3, 84, 84)
robot0_eef_pos               (B, 2, 3)
robot0_eef_quat              (B, 2, 4)
robot0_gripper_qpos          (B, 2, 2)
```

The action target is:

```text
action                       (B, 16, 10)
```

where:

```text
n_obs_steps    = 2
horizon        = 16
n_action_steps = 8
action_dim     = 10
```

The 10D action is an absolute action representation:

```text
position 3 + rotation_6d 6 + gripper 1 = 10
```

The original Robomimic action is 7D:

```text
position 3 + axis_angle 3 + gripper 1 = 7
```

During dataset loading, the 3D axis-angle rotation is converted to a 6D
rotation representation, so the policy learns `(B, 16, 10)`.

## Observation Encoder

Before passing observations to the policy, the time dimension is flattened:

```text
(B, 2, ...) -> (B * 2, ...)
```

Each image stream uses the same architecture, but the two streams do not share
weights.

```text
input image: (B * 2, 3, 84, 84)

CropRandomizer:
  train: random crop
  eval: center crop
  output: (B * 2, 3, 76, 76)

ResNet18Conv:
  conv1 7x7 stride 2, 3 -> 64          (B * 2, 64, 38, 38)
  GroupNorm + ReLU                     (B * 2, 64, 38, 38)
  MaxPool 3x3 stride 2                 (B * 2, 64, 19, 19)
  layer1, 2 BasicBlocks, 64 channels   (B * 2, 64, 19, 19)
  layer2, 2 BasicBlocks, 128 channels  (B * 2, 128, 10, 10)
  layer3, 2 BasicBlocks, 256 channels  (B * 2, 256, 5, 5)
  layer4, 2 BasicBlocks, 512 channels  (B * 2, 512, 3, 3)

SpatialSoftmax:
  1x1 conv 512 -> 32 keypoint maps
  expected (x, y) per keypoint
  output: (B * 2, 32, 2)

Flatten:
  output: (B * 2, 64)

Linear:
  64 -> 64
  output: (B * 2, 64)
```

Per observation timestep, the final feature is:

```text
agentview_image encoding          64
robot0_eye_in_hand_image encoding 64
robot0_eef_pos                     3
robot0_eef_quat                    4
robot0_gripper_qpos                2
-------------------------------------
total                            137
```

Since `n_obs_steps = 2`, the final observation condition is:

```text
global_cond = (B, 137 * 2) = (B, 274)
```

## Noise And Diffusion Input

Noise is part of the diffusion input, but it is not concatenated as an extra
feature dimension. Instead, the action trajectory itself is noised.

During training:

```text
expert action trajectory:
  trajectory = (B, 16, 10)

sampled noise:
  noise = randn_like(trajectory) = (B, 16, 10)

noisy trajectory:
  noise_traj = trajectory + noise * time = (B, 16, 10)
```

During inference:

```text
initial trajectory:
  trajectory = randn((B, 16, 10))
```

So the main U-Net input remains:

```text
sample = (B, 16, 10)
```

Noise changes the values of `sample`, not the `action_dim`.

## CTM U-Net Input And Output

The CTM U-Net receives:

```text
sample/noisy action trajectory: (B, 16, 10)
timestep t:                     (B,)
stop time s:                    (B,)
global_cond from obs encoder:   (B, 274)
```

The timestep and stop-time are each embedded to 128 dimensions:

```text
t embedding:    (B, 128)
s embedding:    (B, 128)
obs condition:  (B, 274)
-------------------------
condition:      (B, 530)
```

The U-Net has:

```text
input_dim       = 10
global_cond_dim = 274
condition width = 530
down_dims       = [512, 1024, 2048]
```

Its forward signature is effectively:

```text
model(sample, timestep, stoptime, global_cond)

input:
  sample      (B, 16, 10)
  condition   (B, 530)

output:
  trajectory  (B, 16, 10)
```

## Policy Output

The model predicts a full horizon action trajectory:

```text
action_pred = (B, 16, 10)
```

The executed action chunk is sliced from that trajectory:

```text
start = n_obs_steps - 1 = 1
end   = start + n_action_steps = 9

action = action_pred[:, 1:9]
action = (B, 8, 10)
```

Before stepping the Robosuite environment, the 6D rotation is converted back to
axis-angle:

```text
(B, 8, 10) -> (B, 8, 7)
```

## Code References

- Policy implementation:
  `sonic_impl/consistency-policy/consistency_policy/student/ctm_policy.py`
- CTM U-Net:
  `sonic_impl/consistency-policy/consistency_policy/ctm_unet.py`
- Noise scheduler:
  `sonic_impl/consistency-policy/consistency_policy/diffusion.py`
- Dataset and action conversion:
  `sonic_impl/consistency-policy/consistency_policy/dataset.py`
- Evaluation action conversion:
  `robomimic_eval/transforms.py`

