# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi

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


        # Domain randomization ranges
        if self.cfg.is_train:
            # wide ranges for training
            self._twr_min = self.cfg.thrust_to_weight * 0.75
            self._twr_max = self.cfg.thrust_to_weight * 1.25

            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.25
            self._k_aero_xy_max = self.cfg.k_aero_xy * 4.0
            self._k_aero_z_min  = self.cfg.k_aero_z * 0.25
            self._k_aero_z_max  = self.cfg.k_aero_z * 4.0

            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.60
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.40
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.60
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.40
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.40
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.60

            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.60
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.40
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.60
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.40
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.40
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.60
        else:
            # narrow ranges for play
            self._twr_min = self.cfg.thrust_to_weight * 0.95
            self._twr_max = self.cfg.thrust_to_weight * 1.05

            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
            self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
            self._k_aero_z_min  = self.cfg.k_aero_z * 0.5
            self._k_aero_z_max  = self.cfg.k_aero_z * 2.0

            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.70
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.30

            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.70
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.30

    def get_rewards(self) -> torch.Tensor:
        """Compute rewards for drone racing through gates with minimal lap time."""

        # # Check cross in gate local frame
        # crossed_gate_plane = self.env._pose_drone_wrt_gate[:, 0] < 0.2
        
        # # Check if drone is within gate shape boundaries
        # within_gate_bounds = (
        #     (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < 0.6) &  # Y within gate width
        #     (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < 0.6)    # Z within gate height
        # )
        
        # # Check if drone was previously behind the gate 
        # was_behind_gate = self.env._prev_x_drone_wrt_gate > 0

        # gate_normals = self.env._normal_vectors[self.env._idx_wp, :]  # (num_envs, 3)
        # vel_along_normal = torch.sum(self.env._robot.data.root_com_lin_vel_w * gate_normals, dim=1)
        # correct_traversal_direction = vel_along_normal < 0

        # gate_passed = crossed_gate_plane & within_gate_bounds & was_behind_gate & correct_traversal_direction

        # wrong_direction_pass = (
        #     crossed_gate_plane &
        #     within_gate_bounds &
        #     ~correct_traversal_direction &   # velocity is in the wrong direction (+X)
        #     (vel_along_normal > 0.5)         # threshold to avoid penalizing near-tangential passes
        # )


        current_x = self.env._pose_drone_wrt_gate[:, 0]
        current_y = self.env._pose_drone_wrt_gate[:, 1]
        current_z = self.env._pose_drone_wrt_gate[:, 2]
        prev_x = self.env._prev_x_drone_wrt_gate

        within_gate_bounds = (
            (torch.abs(current_y) < 0.6) &
            (torch.abs(current_z) < 0.6)
        )

        # correct traversal: + -> -
        gate_passed = (prev_x > 0.0) & (current_x <= 0.0) & within_gate_bounds

        # wrong traversal: - -> +
        backwards_pass = (prev_x < 0.0) & (current_x >= 0.0) & within_gate_bounds

        # self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()

        # give gate passing bonus
        # gate_pass_bonus = gate_passed.float() * 10.0

        # # wrong direction penalty
        # # wrong_direction_penalty = wrong_direction_pass.float()
        # wrong_direction_penalty = backwards_pass.float()

        # # identify_backwards_pass = torch.where(backwards_pass)[0]
        # # if len(identify_backwards_pass) > 0:
        # #     self.env._crashed[identify_backwards_pass] = 200


        gate_pass_bonus = gate_passed.float() * 10.0

        gate_normals = self.env._normal_vectors[self.env._idx_wp, :]
        vel_along_normal = torch.sum(self.env._robot.data.root_com_lin_vel_w * gate_normals, dim=1)

        is_wrong_side = current_x < 0.0
        moving_wrong_way = vel_along_normal > 0.3
        near_gate_opening = (
            (torch.abs(current_y) < 0.8) &
            (torch.abs(current_z) < 0.8)
        )

        wrong_direction_event = is_wrong_side & moving_wrong_way & near_gate_opening
        wrong_direction_penalty = wrong_direction_event.float() + 2.0 * backwards_pass.float()

        num_gates = self.env._waypoints.shape[0]

        # update waypoint for next gate
        # ids_gate_passed = torch.where(gate_passed)[0]
        # if len(ids_gate_passed) > 0:
        #     prev_wp_idx = self.env._idx_wp[ids_gate_passed].clone()

        #     self.env._n_gates_passed[ids_gate_passed] += 1
        #     self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates

        #     # lap detection: from last gate to gate 0
        #     just_completed_lap_mask = (prev_wp_idx == (num_gates - 1))
        #     lap_done_envs = ids_gate_passed[just_completed_lap_mask]
        #     if len(lap_done_envs) > 0:
        #         self._lap_counts[lap_done_envs] += 1

        #     # Update desired positions to next gate
        #     self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        #     self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            
        #     # Update gate-relative pose for new target gate
        #     self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
        #         self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
        #         self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
        #         self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
        #     )


        # Update waypoint for environments that passed the gate
        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:
            prev_wp_idx = self.env._idx_wp[ids_gate_passed].clone()

            self.env._n_gates_passed[ids_gate_passed] += 1
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates

            # lap detection: from last gate to gate 0
            just_completed_lap_mask = (prev_wp_idx == (num_gates - 1))
            lap_done_envs = ids_gate_passed[just_completed_lap_mask]
            if len(lap_done_envs) > 0:
                self._lap_counts[lap_done_envs] += 1

            # Update desired positions to next gate
            new_idx = self.env._idx_wp[ids_gate_passed]
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[new_idx, :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[new_idx, 2]

            # Update gate-relative pose for new target gate
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[new_idx, :3],
                self.env._waypoints_quat[new_idx, :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
            )

            # IMPORTANT:
            # For envs that just switched to a NEW gate, prev_x must be set in the NEW gate frame
            new_gate_pos = self.env._waypoints[new_idx, :3]
            new_gate_quat = self.env._waypoints_quat[new_idx, :]
            new_pose, _ = subtract_frame_transforms(
                new_gate_pos,
                new_gate_quat,
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]

        # For envs that did NOT pass a gate this step, just store current_x
        not_passed_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        not_passed_mask[ids_gate_passed] = False
        self.env._prev_x_drone_wrt_gate[not_passed_mask] = current_x[not_passed_mask]

        distance_to_gate = torch.linalg.norm(
            self.env._desired_pos_w[:, :2] - self.env._robot.data.root_link_pos_w[:, :2], dim=1
        )
        prev_distance_to_gate = self.env._last_distance_to_goal
        progress_raw = prev_distance_to_gate - distance_to_gate
        self.env._last_distance_to_goal = distance_to_gate.detach()
        progress_to_gate = torch.clamp(progress_raw, -1.0, 1.0)

        drone_to_gate_vec = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        drone_to_gate_vec_normalized = drone_to_gate_vec / (distance_to_gate.unsqueeze(1) + 1e-6)

        vel_w = self.env._robot.data.root_com_lin_vel_w
        velocity_towards_gate = torch.sum(vel_w * drone_to_gate_vec_normalized, dim=1)
        velocity_reward = torch.clamp(velocity_towards_gate, -1.0, 6.0)

        euler_tuple = euler_xyz_from_quat(self.env._robot.data.root_quat_w)
        roll = euler_tuple[0]
        pitch = euler_tuple[1]
        
        tilt_penalty = torch.clamp(torch.abs(roll) + torch.abs(pitch) - 0.5, 0.0, 2.0)
        ang_vel_penalty = torch.linalg.norm(self.env._robot.data.root_ang_vel_b, dim=1) * 0.1
        
        # Detect crashes using contact sensor
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        
        crash_penalty = (self.env._crashed > 0).float()


        # Low-altitude penalty: penalize descending too fast when flying too low
        drone_lin_vel_w = self.env._robot.data.root_com_lin_vel_w
        vz_w = drone_lin_vel_w[:, 2]
        downward_speed = torch.clamp(-vz_w - 0.3, min=0.0)

        drone_height = self.env._robot.data.root_link_pos_w[:, 2]
        low_height_mask = (drone_height < 0.35).float()

        downward_velocity_penalty = downward_speed * low_height_mask
        

        # Command penalty: penalize large actions and abrupt action changes
        ctrl_magnitude = torch.linalg.norm(self.env._actions, dim=1)
        ctrl_smoothed = torch.linalg.norm(self.env._actions - self.env._previous_actions, dim=1) ** 2
        ctrl_penalty = 0.0005 * ctrl_magnitude + 0.0002 * ctrl_smoothed

        

        if self.cfg.is_train:
            rewards = {
                "progress_gate": progress_to_gate * self.env.rew['progress_gate_reward_scale'],
                "velocity_forward": velocity_reward * self.env.rew['velocity_forward_reward_scale'],
                "gate_pass": gate_pass_bonus * self.env.rew['gate_pass_reward_scale'],
                "tilt": -tilt_penalty * self.env.rew['tilt_reward_scale'],
                "ang_vel": -ang_vel_penalty * self.env.rew['ang_vel_reward_scale'],
                "crash": -crash_penalty * self.env.rew['crash_reward_scale'],
                # "height": -height_penalty * self.env.rew["height_reward_scale"],
                "wrong_direction": -wrong_direction_penalty * self.env.rew['wrong_direction_reward_scale'],
                "low_altitude": -downward_velocity_penalty * self.env.rew["low_altitude_reward_scale"],
                "ctrl": -ctrl_penalty * self.env.rew["ctrl_reward_scale"],
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

        # position in world frame
        drone_pos_w = self.env._robot.data.root_link_pos_w
        
        # linear velocity in body frame
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        
        # angular velocity in body frame 
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        
        # quaternion
        drone_quat_w = self.env._robot.data.root_quat_w
        
        euler_tuple = euler_xyz_from_quat(drone_quat_w)
        euler_angles = torch.stack(euler_tuple, dim=-1)  # Stack into single tensor
        
        # current target gate index
        current_gate_idx = self.env._idx_wp
        
        # current gate position in world frame
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]

        drone_pos_gate_frame = self.env._pose_drone_wrt_gate
        
        gate_pos_b, gate_quat_b = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w
        )
        
        dist_to_gate = torch.linalg.norm(gate_pos_b, dim=1, keepdim=True)
        
        gate_direction_b = gate_pos_b / (dist_to_gate + 1e-6)
        
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            next_gate_pos_w
        )
        
        gates_passed_normalized = self.env._n_gates_passed.unsqueeze(1).float() / self.env._waypoints.shape[0]
        
        prev_actions = self.env._previous_actions


        # stack into observation vector
        obs = torch.cat(
            [
                drone_pos_w,                    
                drone_lin_vel_b,                
                drone_ang_vel_b,               
                euler_angles,               
                drone_quat_w[:, 3:4],          
                gate_pos_b,                     
                gate_direction_b,               
                dist_to_gate,                   
                drone_pos_gate_frame,           
                next_gate_pos_b,                
                gates_passed_normalized,        
                prev_actions,                   
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

        
        self.env._robot.reset(env_ids)

        # domain randomization
        # if self.cfg.is_train:
        #     # self._randomize_dynamics(env_ids)
        #     self._randomize_domain(env_ids, self.cfg.randomize_domain)
        self._randomize_domain(env_ids, self.cfg.randomize_domain)

        
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

        
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        if self.cfg.is_train:
            
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            
            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]
            
            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)  # 1-3m behind
            y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)   # Lateral variation
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)   # Vertical variation
            

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
            
        else:
            # random position relative to initial waypoint
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

        initial_pos_xy = default_root_state[:, :2]

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - initial_pos_xy, dim=1
        )
        self.env._n_gates_passed[env_ids] = 0
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        # self.env._prev_x_drone_wrt_gate[env_ids] = -1.0  # Initialize behind gate
        self.env._prev_x_drone_wrt_gate[env_ids] = -x_local  # Initialize behind gate


        self.env._crashed[env_ids] = 0


    # def _randomize_dynamics(self, env_ids: torch.Tensor):
    #     """Domain-randomize dynamics for the given env indices (training only)."""
    #     device = self.device
    #     n = len(env_ids)

    #     r = torch.rand(n, device=device)

    #     twr_min = self.cfg.thrust_to_weight * 0.95
    #     twr_max = self.cfg.thrust_to_weight * 1.05
    #     twr_samples = twr_min + r * (twr_max - twr_min)
    #     self.env._thrust_to_weight[env_ids] = twr_samples

    #     r_xy = torch.rand(n, device=device)
    #     r_z  = torch.rand(n, device=device)

    #     k_xy_min = self.cfg.k_aero_xy * 0.5
    #     k_xy_max = self.cfg.k_aero_xy * 2.0
    #     k_z_min  = self.cfg.k_aero_z * 0.5
    #     k_z_max  = self.cfg.k_aero_z * 2.0

    #     k_xy = k_xy_min + r_xy * (k_xy_max - k_xy_min)
    #     k_z  = k_z_min  + r_z  * (k_z_max  - k_z_min)

    #     self.env._K_aero[env_ids, :2] = k_xy.unsqueeze(1)
    #     self.env._K_aero[env_ids, 2]  = k_z

    #     r_kp_rp = torch.rand(n, device=device)
    #     r_ki_rp = torch.rand(n, device=device)
    #     r_kd_rp = torch.rand(n, device=device)

    #     kp_rp_min = self.cfg.kp_omega_rp * 0.85
    #     kp_rp_max = self.cfg.kp_omega_rp * 1.15
    #     ki_rp_min = self.cfg.ki_omega_rp * 0.85
    #     ki_rp_max = self.cfg.ki_omega_rp * 1.15
    #     kd_rp_min = self.cfg.kd_omega_rp * 0.7
    #     kd_rp_max = self.cfg.kd_omega_rp * 1.3

    #     kp_rp = kp_rp_min + r_kp_rp * (kp_rp_max - kp_rp_min)
    #     ki_rp = ki_rp_min + r_ki_rp * (ki_rp_max - ki_rp_min)
    #     kd_rp = kd_rp_min + r_kd_rp * (kd_rp_max - kd_rp_min)

    #     self.env._kp_omega[env_ids, :2] = kp_rp.unsqueeze(1)
    #     self.env._ki_omega[env_ids, :2] = ki_rp.unsqueeze(1)
    #     self.env._kd_omega[env_ids, :2] = kd_rp.unsqueeze(1)

    #     r_kp_y = torch.rand(n, device=device)
    #     r_ki_y = torch.rand(n, device=device)
    #     r_kd_y = torch.rand(n, device=device)

    #     kp_y_min = self.cfg.kp_omega_y * 0.85
    #     kp_y_max = self.cfg.kp_omega_y * 1.15
    #     ki_y_min = self.cfg.ki_omega_y * 0.85
    #     ki_y_max = self.cfg.ki_omega_y * 1.15
    #     kd_y_min = self.cfg.kd_omega_y * 0.7
    #     kd_y_max = self.cfg.kd_omega_y * 1.3

    #     kp_y = kp_y_min + r_kp_y * (kp_y_max - kp_y_min)
    #     ki_y = ki_y_min + r_ki_y * (ki_y_max - ki_y_min)
    #     kd_y = kd_y_min + r_kd_y * (kd_y_max - kd_y_min)

    #     self.env._kp_omega[env_ids, 2] = kp_y
    #     self.env._ki_omega[env_ids, 2] = ki_y
    #     self.env._kd_omega[env_ids, 2] = kd_y


    def _randomize_domain(self, env_ids: torch.Tensor, randomize: bool):
        n = len(env_ids)
        dev = self.device

        def _uniform(low, high):
            return torch.empty(n, device=dev).uniform_(low, high)

        if randomize:
            self.env._thrust_to_weight[env_ids] = _uniform(self._twr_min, self._twr_max)

            self.env._K_aero[env_ids, 0] = _uniform(self._k_aero_xy_min, self._k_aero_xy_max)
            self.env._K_aero[env_ids, 1] = _uniform(self._k_aero_xy_min, self._k_aero_xy_max)
            self.env._K_aero[env_ids, 2] = _uniform(self._k_aero_z_min, self._k_aero_z_max)

            rp_kp = _uniform(self._kp_omega_rp_min, self._kp_omega_rp_max)
            rp_ki = _uniform(self._ki_omega_rp_min, self._ki_omega_rp_max)
            rp_kd = _uniform(self._kd_omega_rp_min, self._kd_omega_rp_max)
            self.env._kp_omega[env_ids, 0] = rp_kp
            self.env._kp_omega[env_ids, 1] = rp_kp
            self.env._ki_omega[env_ids, 0] = rp_ki
            self.env._ki_omega[env_ids, 1] = rp_ki
            self.env._kd_omega[env_ids, 0] = rp_kd
            self.env._kd_omega[env_ids, 1] = rp_kd

            self.env._kp_omega[env_ids, 2] = _uniform(self._kp_omega_y_min, self._kp_omega_y_max)
            self.env._ki_omega[env_ids, 2] = _uniform(self._ki_omega_y_min, self._ki_omega_y_max)
            self.env._kd_omega[env_ids, 2] = _uniform(self._kd_omega_y_min, self._kd_omega_y_max)
        else:
            self.env._thrust_to_weight[env_ids] = self.env._twr_value

            self.env._K_aero[env_ids, 0] = self.env._k_aero_xy_value
            self.env._K_aero[env_ids, 1] = self.env._k_aero_xy_value
            self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value

            self.env._kp_omega[env_ids, 0] = self.env._kp_omega_rp_value
            self.env._kp_omega[env_ids, 1] = self.env._kp_omega_rp_value
            self.env._ki_omega[env_ids, 0] = self.env._ki_omega_rp_value
            self.env._ki_omega[env_ids, 1] = self.env._ki_omega_rp_value
            self.env._kd_omega[env_ids, 0] = self.env._kd_omega_rp_value
            self.env._kd_omega[env_ids, 1] = self.env._kd_omega_rp_value

            self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value
            self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value
            self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value

        self.env._tau_m[env_ids] = self.env._tau_m_value