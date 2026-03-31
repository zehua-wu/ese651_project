# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Lap timing initialization
        self._last_lap_time = torch.zeros(self.num_envs, device=self.device)

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        #Lap Counter
        self._lap_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device = self.device
        )

    def get_rewards(self) -> torch.Tensor:
        """Compute rewards for drone racing through gates with minimal lap time."""

        # ==================== GATE PASSING DETECTION ====================
        # Check if drone has passed through current gate
        # Gate is considered passed when drone crosses the gate plane (x > 0 in gate frame)
        # and is within reasonable lateral bounds
        
        # Distance to gate center in gate frame
        dist_to_gate_center = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        
        # Check if drone crossed the gate plane (positive x in gate frame means passed)
        crossed_gate_plane = self.env._pose_drone_wrt_gate[:, 0] < 0.2
        
        # Check if drone is within gate boundaries (laterally)
        within_gate_bounds = (
            (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < 0.6) &  # Y within gate width
            (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < 0.6)    # Z within gate height
        )
        
        # Check if drone was previously behind the gate (on the approach side, +X in gate frame)
        was_behind_gate = self.env._prev_x_drone_wrt_gate > 0

        # Check traversal direction: drone velocity must have a negative component along the gate
        # normal (i.e. moving from +X side to -X side, against the normal direction).
        # This prevents reverse traversal where the drone passes through the gate in the wrong
        # direction as defined by the powerloop.
        gate_normals = self.env._normal_vectors[self.env._idx_wp, :]  # (num_envs, 3)
        vel_along_normal = torch.sum(self.env._robot.data.root_com_lin_vel_w * gate_normals, dim=1)
        correct_traversal_direction = vel_along_normal < 0

        # Gate is passed when drone crosses plane from the correct approach side, is within bounds,
        # and is moving in the correct direction (against the gate normal)
        gate_passed = crossed_gate_plane & within_gate_bounds & was_behind_gate & correct_traversal_direction

        # Detect wrong-direction traversal: drone physically crossed the gate plane and is within
        # bounds, but came from the wrong side (-X) or is moving in the wrong direction (+X).
        wrong_direction_pass = (
            crossed_gate_plane &
            within_gate_bounds &
            ~correct_traversal_direction &   # velocity is in the wrong direction (+X)
            (vel_along_normal > 0.5)         # threshold to avoid penalizing near-tangential passes
        )

        # Update previous x position for next timestep
        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()

        # Gate passing bonus (large reward for successfully passing through)
        gate_pass_bonus = gate_passed.float() * 10.0

        # Wrong direction penalty
        wrong_direction_penalty = wrong_direction_pass.float()

        # LAP COUNTER
        num_gates = self.env._waypoints.shape[0]
        lap_bonus = torch.zeros(self.num_envs, device=self.device)
        ##########################################

        # LAP TIMER

        # Lap-time reward buffer
        lap_time_reward = torch.zeros(self.num_envs, device=self.device)

        # Current episode time in seconds
        current_time = self.env.episode_length_buf.float() / self.cfg.policy_rate_hz
        
        # Update waypoint indices for environments that passed gates
        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:

            # LAP COUNTER
            # gate index BEFORE increment
            prev_wp_idx = self.env._idx_wp[ids_gate_passed].clone()

            self.env._n_gates_passed[ids_gate_passed] += 1
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates

             # >>> lap detection: from last gate to gate 0
            just_completed_lap_mask = (prev_wp_idx == (num_gates - 1))
            lap_done_envs = ids_gate_passed[just_completed_lap_mask]

            if len(lap_done_envs) > 0:
                self._lap_counts[lap_done_envs] += 1  # strategy-local lap count

                lap_times = current_time[lap_done_envs] - self._last_lap_time[lap_done_envs]

                # Linear reward: target - lap_time
                #   lap_time < 6.2 → positive
                #   lap_time = 6.2 → zero
                #   lap_time > 6.2 → negative
                target_lap_time = 6.0
                lap_time_reward[lap_done_envs] = target_lap_time - lap_times

                # Update last lap time for these envs
                self._last_lap_time[lap_done_envs] = current_time[lap_done_envs]

                # assign a lap bonus (tune this value)
                lap_bonus_value = 30.0
                lap_bonus[lap_done_envs] = lap_bonus_value

            ############################### LAP COUNTER END
            
            # Update desired positions to next gate
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            
            # Update gate-relative pose for new target gate
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
            )

        # ==================== PROGRESS METRICS ====================
        # Distance to current gate
        # distance_to_gate = torch.linalg.norm(
        #     self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        # )
        ################## NEW PROGRESS GATE STARTEGY #######################
        distance_to_gate = torch.linalg.norm(
            self.env._desired_pos_w[:, :2] - self.env._robot.data.root_link_pos_w[:, :2], dim=1
        )
        # Previous distance (stored in env)
        prev_distance_to_gate = self.env._last_distance_to_goal
        # Progress = how much closer you got since last step
        progress_raw = prev_distance_to_gate - distance_to_gate 
        # Update buffer for next step
        self.env._last_distance_to_goal = distance_to_gate.detach()
        # Clamp to avoid huge spikes
        progress_to_gate = torch.clamp(progress_raw, -1.0, 1.0)
        #####################################################################
        
        # Progress reward: inversely proportional to distance
        # Use exponential decay to give stronger reward when close
        # progress_to_gate = torch.exp(-distance_to_gate / 2.0)
        ################## VELOCITY METRIC #######################
        ################## VELOCITY METRIC #######################
        # Velocity towards gate (encourage forward motion)
        drone_to_gate_vec = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        drone_to_gate_vec_normalized = drone_to_gate_vec / (distance_to_gate.unsqueeze(1) + 1e-6)
        
        # Dot product of velocity with direction to gate
        vel_w = self.env._robot.data.root_com_lin_vel_w
        velocity_towards_gate = torch.sum(vel_w * drone_to_gate_vec_normalized, dim=1)
        velocity_reward = torch.clamp(velocity_towards_gate, -1.0, 6.0)  # Encourage speeds up to 6 m/s

        # Extra penalty for moving backwards relative to the current gate
        # backward_speed = torch.clamp(-velocity_towards_gate, min=0.0)  # only when < 0
        # backward_penalty = backward_speed  
        #backward_motion = -torch.clamp(-velocity_towards_gate, 0, 2.0)

        # ==================== GATE 1-2 STRAIGHT SPEED BONUS ====================
        # Encourage high speeds on the long straight between gate 1 and 2 (7m drop)


        speed_bonus = torch.zeros(self.num_envs, device=self.device)  
        if torch.any(self.env._idx_wp == 1) or torch.any(self.env._idx_wp == 2):
            straight_mask = (self.env._idx_wp == 1) | (self.env._idx_wp == 2)
            speed = torch.linalg.norm(vel_w, dim=1)
            
            # Only activate after 2500 iterations
            if self.env.iteration >= 2500:
                # Reward for speeds above 12 m/s, penalty for speeds below
                speed_excess = speed - 12.0  # Positive above 12, negative below 12
                speed_excess = torch.clamp(speed_excess, -6.0, 6.0)  # Limit range: -6 to +6
                
                speed_bonus = speed_excess * straight_mask.float()
    
        
        # ==================== ORIENTATION ALIGNMENT ====================
        # Reward for pointing towards the gate
        drone_forward_w = torch.zeros((self.num_envs, 3), device=self.device)
        drone_forward_w[:, 0] = -1.0  # Forward is +X in body frame
        
        # Rotate forward vector to world frame
        rot_mat = matrix_from_quat(self.env._robot.data.root_quat_w)
        drone_forward_world = torch.bmm(rot_mat, drone_forward_w.unsqueeze(-1)).squeeze(-1)
        
        # Alignment between drone heading and direction to gate
        heading_alignment = torch.sum(drone_forward_world * drone_to_gate_vec_normalized, dim=1)
        heading_reward = torch.clamp(heading_alignment, -1.5, 1.0)
        
        # ==================== STABILITY AND CONTROL ====================
        # Penalize excessive roll and pitch (encourage stable flight)
        euler_tuple = euler_xyz_from_quat(self.env._robot.data.root_quat_w)
        roll = euler_tuple[0]
        pitch = euler_tuple[1]
        
        # Allow some tilt for maneuvering but penalize extreme angles
        # Linear Tilt Penalty
        tilt_penalty = torch.clamp(torch.abs(roll) + torch.abs(pitch) - 0.5, 0.0, 2.0)

        # Quadratic Hinge on Tilt
        # tilt_mag = torch.abs(roll) + torch.abs(pitch)

        # # Free zone up to 0.5 rad
        # tilt_excess = torch.clamp(tilt_mag - 0.6, 0.0)

        # # Quadratic growth after that
        # tilt_penalty = torch.clamp(tilt_excess ** 2, 0.0, 4.0)

        # Speed-aware tilt penalty

        # tilt_mag = torch.abs(roll) + torch.abs(pitch)

        # # Base excess tilt beyond a comfortable threshold
        # tilt_excess = torch.clamp(tilt_mag - 0.4, 0.0, 2.0)  # 0.4 rad free (~23°)

        # # Speed in world frame
        # vel_w = self.env._robot.data.root_com_lin_vel_w
        # speed = torch.linalg.norm(vel_w, dim=1)

        # # Normalize speed: 0 m/s -> 1,  v >= 6 m/s -> ~0
        # speed_norm = torch.clamp(speed / 6.0, 0.0, 1.0)
        # speed_factor = 1.0 - speed_norm

        # tilt_penalty = tilt_excess * speed_factor
                
        # Penalize excessive angular velocities (encourage smooth control)
        ang_vel_penalty = torch.linalg.norm(self.env._robot.data.root_ang_vel_b, dim=1) * 0.1
        
        # ==================== CRASH DETECTION ====================
        # Detect crashes using contact sensor
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        
        # Only count as crashed after initial settling period
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        
        # Large penalty for crashing
        crash_penalty = (self.env._crashed > 0).float()
        
        # ==================== HEIGHT MAINTENANCE ====================
        # Encourage staying near gate height
        target_height = self.env._desired_pos_w[:, 2]
        current_height = self.env._robot.data.root_link_pos_w[:, 2]
        height_error = torch.abs(current_height - target_height)
        height_penalty = torch.clamp(height_error, 0.0, 2.0)

        # ========================= LAP TIME =========================
        # Giving penalty for Per-step time cost, Add a small negative reward every step.
        step_penalty = -0.001  # tiny

        # ==================== COMPUTE FINAL REWARD ====================
        if self.cfg.is_train:
            rewards = {
                "progress_gate": progress_to_gate * self.env.rew['progress_gate_reward_scale'],
                "velocity_forward": velocity_reward * self.env.rew['velocity_forward_reward_scale'],
                "gate_pass": gate_pass_bonus * self.env.rew['gate_pass_reward_scale'],
                # "heading_alignment": heading_reward * self.env.rew['heading_alignment_reward_scale'],
                "tilt": -tilt_penalty * self.env.rew['tilt_reward_scale'],
                "ang_vel": -ang_vel_penalty * self.env.rew['ang_vel_reward_scale'],
                "crash": -crash_penalty * self.env.rew['crash_reward_scale'],
                "wrong_direction": -wrong_direction_penalty * self.env.rew['wrong_direction_reward_scale'],
                # "height": -height_penalty * self.env.rew['height_reward_scale'],
                # "backward": backward_motion * self.env.rew['backward_reward_scale']
                # "step": step_penalty * self.env.rew["step_reward_scale"],
            }
            
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Apply death cost for terminated episodes
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward
            )

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for the racing policy."""

        # ==================== DRONE STATE ====================
        # Position in world frame
        drone_pos_w = self.env._robot.data.root_link_pos_w
        
        # Linear velocity in body frame (more intuitive for control)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        
        # Angular velocity in body frame (body rates: roll, pitch, yaw rates)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        
        # Orientation as quaternion
        drone_quat_w = self.env._robot.data.root_quat_w
        
        # Euler angles (roll, pitch, yaw) for easier interpretation
        # Note: euler_xyz_from_quat returns a tuple of 3 tensors
        euler_tuple = euler_xyz_from_quat(drone_quat_w)
        euler_angles = torch.stack(euler_tuple, dim=-1)  # Stack into single tensor
        
        # ==================== CURRENT GATE INFORMATION ====================
        # Current target gate index
        current_gate_idx = self.env._idx_wp
        
        # Current gate position in world frame
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]
        
        # Current gate yaw orientation
        current_gate_yaw = self.env._waypoints[current_gate_idx, -1]
        
        # Relative position to current gate in gate frame
        # This provides position relative to the gate's coordinate system
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate
        
        # Relative position to current gate in body frame
        # This tells the drone where the gate is relative to its own orientation
        gate_pos_b, gate_quat_b = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w
        )
        
        # Distance to current gate
        dist_to_gate = torch.linalg.norm(gate_pos_b, dim=1, keepdim=True)
        
        # Normalized direction to gate in body frame
        gate_direction_b = gate_pos_b / (dist_to_gate + 1e-6)
        
        # ==================== NEXT GATE INFORMATION ====================
        # Look ahead to next gate for better trajectory planning
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        
        # Next gate position in body frame
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            next_gate_pos_w
        )
        
        # ==================== PROGRESS INFORMATION ====================
        # Number of gates passed (normalized)
        gates_passed_normalized = self.env._n_gates_passed.unsqueeze(1).float() / self.env._waypoints.shape[0]
        
        # ==================== PREVIOUS ACTIONS ====================
        # Include previous actions to help with temporal consistency
        prev_actions = self.env._previous_actions

        # ==================== ASSEMBLE OBSERVATION VECTOR ====================
        obs = torch.cat(
            [
                # Drone state (13 dims)
                drone_pos_w,                    # Position in world (3)
                drone_lin_vel_b,                # Linear velocity in body (3)
                drone_ang_vel_b,                # Angular velocity in body (3)
                euler_angles,                   # Roll, pitch, yaw (3)
                drone_quat_w[:, 3:4],          # Quaternion w component (1) - for full orientation
                
                # Current gate relative information (10 dims)
                gate_pos_b,                     # Gate position in body frame (3)
                gate_direction_b,               # Normalized direction to gate (3)
                dist_to_gate,                   # Distance to gate (1)
                drone_pos_gate_frame,           # Position in gate frame (3)
                
                # Next gate information (3 dims)
                next_gate_pos_b,                # Next gate position in body frame (3)
                
                # Progress and history (5 dims)
                gates_passed_normalized,        # Progress through course (1)
                prev_actions,                   # Previous actions (4)
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states with randomization."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # >>> Domain randomization: TRAINING ONLY <<<
        if self.cfg.is_train:
            self._randomize_dynamics(env_ids)

        # >>> Domain randomization: TRAINING ONLY <<<
        if self.cfg.is_train:
            self._randomize_dynamics(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # ==================== TRAINING MODE RESET ====================
        if self.cfg.is_train:
            # Start from random waypoints for curriculum learning
            # Early in training, start from earlier gates; later, randomize more
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            
            # Get starting gate information
            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]
            
            # Randomize starting position relative to gate
            # Position behind the gate with some variation
            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -1.0)  # 1-3m behind
            y_local = torch.empty(n_reset, device=self.device).uniform_(-0.8, 0.8)   # Lateral variation
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)   # Vertical variation
            
            # Rotate local position to global frame
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = z_local + z_wp
            
            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            
            # Point drone towards the gate with some random yaw offset
            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            # initial_yaw = self.env._waypoints[waypoint_indices, -1]

            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)  # ±17 degrees
            
            quat = quat_from_euler_xyz(
                torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1),  # Small roll variation
                torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1),  # Small pitch variation
                initial_yaw + yaw_noise
            )
            default_root_state[:, 3:7] = quat
            
            # Add small initial velocity towards gate for more dynamic starts
            initial_speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 0.5)
            vel_direction = torch.stack([
                torch.cos(initial_yaw + yaw_noise),
                torch.sin(initial_yaw + yaw_noise),
                torch.zeros(n_reset, device=self.device)
            ], dim=1)
            default_root_state[:, 7:10] = vel_direction * initial_speed.unsqueeze(1)
            
        # ==================== PLAY MODE RESET ====================
        else:
            # Play mode: random position relative to initial waypoint
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # Rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # Point drone towards the gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        # self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
        #     self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        # )

        initial_pos_xy = default_root_state[:, :2]

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - initial_pos_xy, dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = -1.0  # Initialize behind gate

        self.env._crashed[env_ids] = 0

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        """Domain-randomize dynamics for the given env indices (training only)."""
        device = self.device
        n = len(env_ids)

        # Uniform in [0,1] for each sampled group
        r = torch.rand(n, device=device)

        # ----- TWR -----
        # factors: 0.95 to 1.05
        twr_min = self.cfg.thrust_to_weight * 0.95
        twr_max = self.cfg.thrust_to_weight * 1.05
        twr_samples = twr_min + r * (twr_max - twr_min)
        self.env._thrust_to_weight[env_ids] = twr_samples

        # ----- Aerodynamics -----
        r_xy = torch.rand(n, device=device)
        r_z  = torch.rand(n, device=device)

        k_xy_min = self.cfg.k_aero_xy * 0.5
        k_xy_max = self.cfg.k_aero_xy * 2.0
        k_z_min  = self.cfg.k_aero_z * 0.5
        k_z_max  = self.cfg.k_aero_z * 2.0

        k_xy = k_xy_min + r_xy * (k_xy_max - k_xy_min)
        k_z  = k_z_min  + r_z  * (k_z_max  - k_z_min)

        self.env._K_aero[env_ids, :2] = k_xy.unsqueeze(1)
        self.env._K_aero[env_ids, 2]  = k_z

        # ----- PID roll/pitch gains -----
        r_kp_rp = torch.rand(n, device=device)
        r_ki_rp = torch.rand(n, device=device)
        r_kd_rp = torch.rand(n, device=device)

        kp_rp_min = self.cfg.kp_omega_rp * 0.85
        kp_rp_max = self.cfg.kp_omega_rp * 1.15
        ki_rp_min = self.cfg.ki_omega_rp * 0.85
        ki_rp_max = self.cfg.ki_omega_rp * 1.15
        kd_rp_min = self.cfg.kd_omega_rp * 0.7
        kd_rp_max = self.cfg.kd_omega_rp * 1.3

        kp_rp = kp_rp_min + r_kp_rp * (kp_rp_max - kp_rp_min)
        ki_rp = ki_rp_min + r_ki_rp * (ki_rp_max - ki_rp_min)
        kd_rp = kd_rp_min + r_kd_rp * (kd_rp_max - kd_rp_min)

        self.env._kp_omega[env_ids, :2] = kp_rp.unsqueeze(1)
        self.env._ki_omega[env_ids, :2] = ki_rp.unsqueeze(1)
        self.env._kd_omega[env_ids, :2] = kd_rp.unsqueeze(1)

        # ----- PID yaw gains -----
        r_kp_y = torch.rand(n, device=device)
        r_ki_y = torch.rand(n, device=device)
        r_kd_y = torch.rand(n, device=device)

        kp_y_min = self.cfg.kp_omega_y * 0.85
        kp_y_max = self.cfg.kp_omega_y * 1.15
        ki_y_min = self.cfg.ki_omega_y * 0.85
        ki_y_max = self.cfg.ki_omega_y * 1.15
        kd_y_min = self.cfg.kd_omega_y * 0.7
        kd_y_max = self.cfg.kd_omega_y * 1.3

        kp_y = kp_y_min + r_kp_y * (kp_y_max - kp_y_min)
        ki_y = ki_y_min + r_ki_y * (ki_y_max - ki_y_min)
        kd_y = kd_y_min + r_kd_y * (kd_y_max - kd_y_min)

        self.env._kp_omega[env_ids, 2] = kp_y
        self.env._ki_omega[env_ids, 2] = ki_y
        self.env._kd_omega[env_ids, 2] = kd_y
