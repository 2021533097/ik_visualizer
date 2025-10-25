import argparse
import os
import time
from typing import Optional, Union

import numpy as np
import torch
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf

from wrapper_solver import IKSolverWrapper
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import load_yaml, join_path, get_robot_configs_path
from utils import RobotModel, create_grid_of_poses

class PointIKApp:
    def __init__(self, robot_file: str):
        self.server = viser.ViserServer()
        self.robot = RobotModel(self.server, robot_file)
        self.ik_params = {
            "pos_tol": 0.005,
            "rot_tol": 0.05,
            "num_seeds": 20,
            "self_collision_check": True,
            "self_collision_opt": False,
            "use_cuda_graph": False,
        }
        # 初始化 IK SolverWrapper
        self._reload_ik_wrapper()

        # GUI + 控件
        self._setup_gui()
        # 绘制初始 grid（或点云）以便可视化
        self._draw_grid()

        self.run_loop()

    def _reload_ik_wrapper(self):
        # 从 ik_params 重建 IKSolverWrapper
        self.ik_wrapper = IKSolverWrapper(
            self.robot.robot_cfg,
            position_threshold=self.ik_params["pos_tol"],
            rotation_threshold=self.ik_params["rot_tol"],
            num_seeds=self.ik_params["num_seeds"],
            self_collision_check=self.ik_params["self_collision_check"],
            self_collision_opt=self.ik_params["self_collision_opt"],
            use_cuda_graph=self.ik_params["use_cuda_graph"],
        )

    def _setup_gui(self):
        self.server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
        # 现有的 range 控件
        self.ms_x = self.server.gui.add_multi_slider("x_range",
                                                     min=-2.0, max=2.0, step=0.01,
                                                     initial_value=(-0.5, 0.5))
        self.ms_y = self.server.gui.add_multi_slider("y_range",
                                                     min=-2.0, max=2.0, step=0.01,
                                                     initial_value=(-0.5, 0.5))
        self.ms_z = self.server.gui.add_multi_slider("z_range",
                                                     min=-0.5, max=2.0, step=0.01,
                                                     initial_value=(0.2, 0.6))
        self.sl_res = self.server.gui.add_slider("res", min=0.02, max=0.3, step=0.01,
                                                 initial_value=0.1)

        # **新增 IK 参数 控件**
        self.sl_pos = self.server.gui.add_slider("pos_tol", min=0.0, max=0.1, step=0.001,
                                                 initial_value=self.ik_params["pos_tol"])
        self.sl_rot = self.server.gui.add_slider("rot_tol", min=0.0, max=0.5, step=0.005,
                                                 initial_value=self.ik_params["rot_tol"])
        self.sl_seeds = self.server.gui.add_slider("num_seeds", min=1, max=200, step=1,
                                                   initial_value=self.ik_params["num_seeds"])
        self.cb_selfcol = self.server.gui.add_checkbox("self_collision_check",
                                                       initial_value=self.ik_params["self_collision_check"])
        self.cb_selfopt = self.server.gui.add_checkbox("self_collision_opt",
                                                       initial_value=self.ik_params["self_collision_opt"])
        self.cb_cuda = self.server.gui.add_checkbox("use_cuda_graph",
                                                    initial_value=self.ik_params["use_cuda_graph"])
        # “Reload IK 参数”按钮
        self.btn_reload = self.server.gui.add_button("Reload IK Params")
        @self.btn_reload.on_click
        def on_reload(evt):
            # 从 GUI 控件值里读取，更新 ik_params，然后重建 IKWrapper
            self.ik_params["pos_tol"] = float(self.sl_pos.value)
            self.ik_params["rot_tol"] = float(self.sl_rot.value)
            self.ik_params["num_seeds"] = int(self.sl_seeds.value)
            self.ik_params["self_collision_check"] = bool(self.cb_selfcol.value)
            self.ik_params["self_collision_opt"] = bool(self.cb_selfopt.value)
            self.ik_params["use_cuda_graph"] = bool(self.cb_cuda.value)
            self._reload_ik_wrapper()
            print("Reloaded IK params:", self.ik_params)

        # Solve IK 的按钮
        self.btn_solve = self.server.gui.add_button("Solve IK")
        @self.btn_solve.on_click
        def on_click(evt):
            self._solve()

    def _draw_grid(self):
        qu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        x0, x1 = self.ms_x.value
        y0, y1 = self.ms_y.value
        z0, z1 = self.ms_z.value
        res = float(self.sl_res.value)
        self.grid_poses = create_grid_of_poses(x0, x1, y0, y1, z0, z1, res, qu)
        pts = self.grid_poses.position.cpu().numpy()
        colors = np.tile(np.array([0, 0, 255], dtype=np.uint8), (pts.shape[0], 1))
        # 这里我直接 add 点云，你可以改为 scene 管理
        self.server.scene.add_point_cloud(name="/reach_pts", points=pts, colors=colors, point_size=0.02)

    def _solve(self):
        if not hasattr(self, "grid_poses"):
            self._draw_grid()
        success, sols = self.ik_wrapper.solve(self.grid_poses)
        success = success.reshape(-1)
        pts = self.grid_poses.position.cpu().numpy()
        colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
        colors[success, :] = np.array([0, 255, 0], dtype=np.uint8)
        colors[~success, :] = np.array([255, 0, 0], dtype=np.uint8)
        # 更新点云：移除旧的，然后 add 新的
        try:
            self.server.scene.remove("/reach_pts")
        except Exception:
            pass
        self.server.scene.add_point_cloud(name="/reach_pts", points=pts, colors=colors, point_size=0.02)

        # 如果有解，则用第一个解驱动机器人
        idxs = np.nonzero(success)[0]
        if len(idxs) > 0:
            sol0 = sols[idxs[0]]
            sol_tensor = torch.tensor(sol0, dtype=torch.float32)
            self.robot.set_joint_state(sol_tensor)

    def run_loop(self):
        while True:
            time.sleep(0.1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--robot_file', type=str, default='franka.yml')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    app = PointIKApp(args.robot_file)

if __name__ == "__main__":
    main()
