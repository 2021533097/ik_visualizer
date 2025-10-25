import argparse
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.geom.types import WorldConfig
from curobo.util_file import load_yaml, join_path, get_robot_configs_path, get_assets_path

# =========================
# ✅ CONFIG
# =========================
MAX_BATCH = int(1024 * 4)


# ==================================================
# ✅ Argument Parsing
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda_device', type=int, default=0,
        help='CUDA device ID to use (default: 0)'
    )
    parser.add_argument(
        '--robot_file', type=str, default='franka.yml',
        help='Robot config YAML file in cuRobo (default: franka.yml)'
    )
    return parser.parse_args()


# ==================================================
# ✅ CPU-ONLY Pose Grid Generation
# ==================================================
def create_grid_of_poses(x0, x1, y0, y1, z0, z1, resolution,
                         quaternion_unit, tensor_args: TensorDeviceType=TensorDeviceType(device='cpu')) -> Pose:
    """
    在 CPU 上直接构造一个目标网格 Pose 批（位置 + 四元数）。
    """
    def arange_stable(a0, a1, step):
        n = int(round((a1 - a0) / step))
        # 🔸 强制在 CPU 上生成
        return torch.linspace(a0, a1, n + 1, device="cpu", dtype=torch.float32)

    xs = arange_stable(x0, x1, resolution)
    ys = arange_stable(y0, y1, resolution)
    zs = arange_stable(z0, z1, resolution)

    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)  # (N, 3)

    # 四元数单位阵也放在 CPU
    quats = quaternion_unit.cpu().unsqueeze(0).repeat(positions.shape[0], 1)

    return Pose(positions, quats)


# ==================================================
# ✅ Load robot configuration
# ==================================================
def load_robot(robot_file: str) -> RobotConfig:
    cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_cfg = RobotConfig.from_dict(cfg["robot_cfg"])
    return robot_cfg


# ==================================================
# ✅ 防抖（Debouncer）
# ==================================================
class Debouncer:
    def __init__(self, interval_sec: float, func):
        self.interval = interval_sec
        self.func = func
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def call(self, *args, **kwargs):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.interval, self.func, args=args, kwargs=kwargs)
            self._timer.daemon = True
            self._timer.start()


# ==================================================
# ✅ IK Solver Wrapper with CPU-GPU batch transfer
# ==================================================
class IKSolverWrapper:
    def __init__(
        self,
        robot_cfg,
        position_threshold: float = 0.01,
        rotation_threshold: float = 0.1,
        num_seeds: int = 20,
        self_collision_check: bool = True,
        self_collision_opt: bool = False,
        use_cuda_graph: bool = False,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        world_model: Optional[
            Union[Union[List[Dict], List[WorldConfig]], Union[Dict, WorldConfig]]
        ] = None,
    ):
        self.tensor_args = tensor_args
        self.use_cuda_graph = use_cuda_graph

        # 初始化 IK solver config
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

        # 创建 solver
        self.solver = IKSolver(self.ik_config)

    def _pad_pose(self, sub_goal: Pose, fixed_batch: int):
        """将 sub_goal pad 到固定 batch 大小"""
        cur_bs = sub_goal.position.shape[0]
        if cur_bs == fixed_batch:
            return sub_goal, cur_bs

        pad_size = fixed_batch - cur_bs
        device = sub_goal.position.device
        dtype = sub_goal.position.dtype

        pad_pos = torch.zeros((pad_size, 3), device=device, dtype=dtype)
        pad_quat = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).repeat(pad_size, 1)

        padded_pose = Pose(
            torch.cat([sub_goal.position, pad_pos], dim=0),
            torch.cat([sub_goal.quaternion, pad_quat], dim=0)
        )
        return padded_pose, cur_bs

    def solve(self, goal_poses: Pose):
        """
        对一批目标姿态 goal_poses 进行 IK 求解。
        返回:
            success_all: np.ndarray, shape (N,)
            solutions_all: np.ndarray, shape (N, dof)
        """
        # ⚠️ 不直接搬整个 goal_poses 到 GPU，保持在 CPU。
        batch_N = goal_poses.position.shape[0]
        success_list, sol_list = [], []

        for i in range(0, batch_N, MAX_BATCH):
            i_end = min(i + MAX_BATCH, batch_N)
            print(f"Processing IK batch {i} to {i_end} / {batch_N}")

            # 🔸 仅搬当前子 batch 到 GPU
            sub_goal = Pose(
                goal_poses.position[i:i_end].to(**self.tensor_args.as_torch_dict()),
                goal_poses.quaternion[i:i_end].to(**self.tensor_args.as_torch_dict()),
            )

            if self.use_cuda_graph:
                sub_goal, valid_bs = self._pad_pose(sub_goal, MAX_BATCH)
            else:
                valid_bs = sub_goal.position.shape[0]

            # 求解
            res = self.solver.solve_batch(sub_goal)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 截取有效部分
            success_list.append(res.success[:valid_bs].detach().cpu())
            sol_list.append(res.solution[:valid_bs].detach().cpu())

            # ✅ 主动释放 GPU 内存
            del sub_goal, res
            torch.cuda.empty_cache()

        success_all = torch.cat(success_list, dim=0).numpy()
        solutions_all = torch.cat(sol_list, dim=0).numpy()

        return success_all, solutions_all


# ==================================================
# ✅ Robot Wrapper
# ==================================================
class Robot:
    def __init__(self, robot_file: str, tensor_args=TensorDeviceType(), load_collision_meshes: bool=False):
        self.tensor_args = tensor_args
        self.robot_cfg = load_robot(robot_file)
        self.kinematics = self.robot_cfg.kinematics
        self.asset_path = self.kinematics.generator_config.asset_root_path
        self.urdf_path = self.kinematics.generator_config.urdf_path
        self.mesh_path = join_path(self.asset_path, "meshes")

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        print(f"Loading URDF from: {self.urdf_path}, {self.mesh_path}")

        if load_collision_meshes:
            self.urdf_obj = URDF.load(self.urdf_path, mesh_dir=self.mesh_path, build_collision_scene_graph=True, load_collision_meshes=True)
        else:
            self.urdf_obj = URDF.load(self.urdf_path, mesh_dir=self.mesh_path)

    def solve_ik(self, goal_poses: Pose, **ik_kwargs):
        # goal_poses is kept on world space
        ik_solver = IKSolverWrapper(
            self.robot_cfg,
            tensor_args=self.tensor_args,
            **ik_kwargs
        )
        return ik_solver.solve(goal_poses)

    def get_viser_urdf(self, server: viser.ViserServer) -> ViserUrdf:
        vise_urdf = ViserUrdf(
            server,
            urdf_or_path=self.urdf_obj,
            load_meshes=True,
            load_collision_meshes=False,
        )
        return vise_urdf


# ==================================================
# ✅ Reachability Visualizer
# ==================================================
class ReachabilityVisualizer:
    def __init__(self, robot_file: str):
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

        self.robot_cfg = load_robot(robot_file)
        self.tensor_args = TensorDeviceType()
        self.robot = Robot(robot_file, self.tensor_args, load_collision_meshes=False)

        # GUI 控件（略，保持原样）
        self.ms_x = self.server.gui.add_multi_slider("x_range", min=-2.0, max=2.0, step=0.01, initial_value=(-0.5, 0.5))
        self.ms_y = self.server.gui.add_multi_slider("y_range", min=-2.0, max=2.0, step=0.01, initial_value=(-0.5, 0.5))
        self.ms_z = self.server.gui.add_multi_slider("z_range", min=-0.5, max=2, step=0.01, initial_value=(0.2, 0.8))
        self.sl_res = self.server.gui.add_slider("resolution", min=0.02, max=0.2, step=0.01, initial_value=0.1)
        self.sl_pos = self.server.gui.add_slider("pos_tol", min=0.0, max=0.1, step=0.001, initial_value=0.005)
        self.sl_rot = self.server.gui.add_slider("rot_tol", min=0.0, max=0.5, step=0.005, initial_value=0.05)
        self.sl_seeds = self.server.gui.add_slider("num_seeds", min=1, max=256, step=1, initial_value=20)
        self.cb_self = self.server.gui.add_checkbox("self_collision_check", initial_value=True)
        self.cb_self_opt = self.server.gui.add_checkbox("self_collision_opt", initial_value=False)
        self.cb_cuda = self.server.gui.add_checkbox("use_cuda_graph", initial_value=False)
        self.btn_solve = self.server.gui.add_button("Solve")
        self.txt = self.server.gui.add_text("Success", initial_value="0 / 0 (0.0%)")
        
        #为robot的各个link添加可视化
        self.vr = self.robot.get_viser_urdf(self.server)

        # 防抖重建
        self._debounce_rebuild = Debouncer(0.25, self._rebuild_grid_and_paint_blue)
        for handle in [self.ms_x, self.ms_y, self.ms_z, self.sl_res,
                       self.sl_pos, self.sl_rot, self.sl_seeds, self.cb_self, self.cb_self_opt, self.cb_cuda]:
            @handle.on_update
            def _(evt, self=self):
                self._debounce_rebuild.call()

        @self.btn_solve.on_click
        def _(event):
            self._trigger_solve()

        self.pc_handle = None
        self.goal_poses = None
        self._rebuild_grid_and_paint_blue()

        while True:
            time.sleep(0.1)

    def _rebuild_grid_and_paint_blue(self):
        quaternion_unit = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cpu", dtype=torch.float32)

        x0, x1 = self.ms_x.value
        y0, y1 = self.ms_y.value
        z0, z1 = self.ms_z.value
        resolution = float(self.sl_res.value)

        self.goal_poses = create_grid_of_poses(x0, x1, y0, y1, z0, z1, resolution, quaternion_unit, self.tensor_args)
        pos_np = self.goal_poses.position.numpy()
        colors = np.zeros((pos_np.shape[0], 3), dtype=np.uint8)
        colors[:, :] = np.array([0, 0, 255], dtype=np.uint8)

        if self.pc_handle is None:
            self.pc_handle = self.server.scene.add_point_cloud("/reach_pts", points=pos_np, colors=colors, point_size=0.02)
        else:
            self.pc_handle.points = pos_np
            self.pc_handle.colors = colors

        self.txt.value = f"0 / {pos_np.shape[0]} (0.0%)"

    def _trigger_solve(self):
        if self.goal_poses is None:
            return

        def task():
            pos_tol = float(self.sl_pos.value)
            rot_tol = float(self.sl_rot.value)
            num_seeds = int(self.sl_seeds.value)
            use_cuda_graph = bool(self.cb_cuda.value)
            self_collision_check = bool(self.cb_self.value)
            self_collision_opt = bool(self.cb_self_opt.value)

            success, _ = self.robot.solve_ik(
                self.goal_poses,
                position_threshold=pos_tol,
                rotation_threshold=rot_tol,
                num_seeds=num_seeds,
                self_collision_check=self_collision_check,
                self_collision_opt=self_collision_opt,
                use_cuda_graph=use_cuda_graph,
            )

            success = success.reshape(-1)
            
            total = len(success)
            count = int(success.sum())
            rate = count / total if total > 0 else 0.0

            colors = np.zeros((total, 3), dtype=np.uint8)
            colors[success, :] = np.array([0, 255, 0], dtype=np.uint8)
            colors[~success, :] = np.array([255, 0, 0], dtype=np.uint8)

            self.pc_handle.colors = colors
            self.txt.value = f"{count} / {total} ({rate * 100:.1f}%)"
            print(f"Recomputed reachability: {count} / {total} ({rate * 100:.1f}%)")

        thread = threading.Thread(target=task)
        thread.daemon = True
        thread.start()


# ==================================================
# ✅ Entrypoint
# ==================================================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    ReachabilityVisualizer(robot_file=args.robot_file)


if __name__ == "__main__":
    main()
