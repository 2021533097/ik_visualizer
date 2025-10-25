import numpy as np
import os
import torch

import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf

from config import MAX_BATCH
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import load_yaml, join_path, get_robot_configs_path


def load_robot(robot_file: str) -> RobotConfig:
    cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_cfg = RobotConfig.from_dict(cfg["robot_cfg"])
    return robot_cfg

def create_grid_of_poses(x0, x1, y0, y1, z0, z1, resolution,
                         quaternion_unit: torch.Tensor) -> Pose:
    def arange_stable(a0, a1, step):
        n = int(round((a1 - a0) / step))
        return torch.linspace(a0, a1, n + 1, device="cpu", dtype=torch.float32)

    xs = arange_stable(x0, x1, resolution)
    ys = arange_stable(y0, y1, resolution)
    zs = arange_stable(z0, z1, resolution)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    quats = quaternion_unit.cpu().unsqueeze(0).repeat(positions.shape[0], 1)
    return Pose(positions, quats)

class IKSolverWrapper:
    def __init__(self,
                 robot_cfg: RobotConfig,
                 position_threshold: float = 0.01,
                 rotation_threshold: float = 0.1,
                 num_seeds: int = 20,
                 self_collision_check: bool = True,
                 self_collision_opt: bool = False,
                 use_cuda_graph: bool = False,
                 tensor_args: TensorDeviceType = TensorDeviceType()):
        self.tensor_args = tensor_args
        self.use_cuda_graph = use_cuda_graph
        self.ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            use_cuda_graph=use_cuda_graph,
            tensor_args=tensor_args,
        )
        self.solver = IKSolver(self.ik_config)

    def _pad_pose(self, sub: Pose, fixed_batch: int):
        cur_bs = sub.position.shape[0]
        if cur_bs == fixed_batch:
            return sub, cur_bs
        pad = fixed_batch - cur_bs
        device = sub.position.device
        dtype = sub.position.dtype
        pad_pos = torch.zeros((pad, 3), device=device, dtype=dtype)
        pad_quat = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).repeat(pad, 1)
        padded = Pose(
            torch.cat([sub.position, pad_pos], dim=0),
            torch.cat([sub.quaternion, pad_quat], dim=0),
        )
        return padded, cur_bs

    def solve(self, goal_poses: Pose):
        N = goal_poses.position.shape[0]
        success_list = []
        sol_list = []
        for i in range(0, N, MAX_BATCH):
            print(f"Solving IK batch {i} to {min(i + MAX_BATCH, N)} / {N}")
            i_end = min(i + MAX_BATCH, N)
            sub = Pose(
                goal_poses.position[i:i_end].to(device=self.tensor_args.device),
                goal_poses.quaternion[i:i_end].to(device=self.tensor_args.device),
            )
            if self.use_cuda_graph:
                sub, valid_bs = self._pad_pose(sub, MAX_BATCH)
            else:
                valid_bs = sub.position.shape[0]
            res = self.solver.solve_batch(sub)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            success_list.append(res.success[:valid_bs].detach().cpu())
            sol_list.append(res.solution[:valid_bs].detach().cpu())
        success_all = torch.cat(success_list, dim=0).numpy()
        solutions_all = torch.cat(sol_list, dim=0).numpy()
        return success_all, solutions_all

class RobotModel:
    def __init__(self, server: viser.ViserServer, robot_file: str):
        self.robot_cfg = load_robot(robot_file)
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)

        asset_root = self.robot_cfg.kinematics.generator_config.asset_root_path
        urdf_path = self.robot_cfg.kinematics.generator_config.urdf_path
        mesh_dir = os.path.join(asset_root, "meshes")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.urdf_obj = URDF.load(urdf_path, mesh_dir=mesh_dir)
        self.server = server
        if server is not None:
            self.viser_urdf = ViserUrdf(server, urdf_or_path=self.urdf_obj,
                                        load_meshes=True, load_collision_meshes=False)

    def get_full_state(self, joint_angles: torch.Tensor) -> JointState:
        js_active = JointState(position=joint_angles,
                               joint_names=self.robot_cfg.cspace.joint_names)
        js_full = self.kin_model.get_full_js(js_active)
        return js_full

    def set_joint_state(self, joint_angles: torch.Tensor):
        """Set the joint state of the robot.

        Args:
            joint_angles (torch.Tensor): The joint angles to set. Shape (num_joints,) or (1, num_joints).
        """
        js_full = self.get_full_state(joint_angles)
        full = js_full.position.detach().cpu().numpy().flatten()
        try:
            self.visualize_robot(full)
        except Exception as e:
            print("Warning in update_cfg:", e)
    
    def visualize_robot(self, position):
        if self.server is not None:
            self.viser_urdf.update_cfg(position)