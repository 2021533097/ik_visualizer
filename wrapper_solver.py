import numpy as np
import torch

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import load_yaml, join_path, get_robot_configs_path


# ========== IK 封装 ==========
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
    
    def solve_one(self, goal_pose: Pose):
        assert goal_pose.position.shape[0] == 1
        success, solutions = self.solve(goal_pose)
        return success[0], solutions[0]
