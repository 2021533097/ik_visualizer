import argparse
import os
import time
import threading
from typing import Optional

import numpy as np
import torch
import viser

from utils import RobotModel, create_grid_of_poses, IKSolverWrapper
from curobo.types.math import Pose


# 防抖
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


class PointIKApp:
    def __init__(self, robot_file: str):
        self.server = viser.ViserServer()
        self.robot = RobotModel(self.server, robot_file)

        # 默认 IK 参数
        self.ik_params = {
            "pos_tol": 0.005,
            "rot_tol": 0.05,
            "num_seeds": 20,
            "self_collision_check": True,
            "self_collision_opt": False,
            "use_cuda_graph": False,
        }

        self.ik_wrapper = None
        self._reload_ik_wrapper()

        # 点云 / 网格 pose 缓存
        self.pc_handle = None
        self.grid_poses = None
        self.point_state = None  # -1: 未求解, 0: fail, 1: success
        self.point_sol = None  # 保存每点对应的解（如果有的话）

        # 目标物体控制器及初始 pose
        # 你可以修改下面的初始位置 / 四元数
        self.obj_init_pos = (0.5, 0.0, 0.5)
        self.obj_init_wxyz = (1.0, 0.0, 0.0, 0.0)  # 单位四元数
        self.target_obj_pose = (self.obj_init_pos, self.obj_init_wxyz)
        self.obj_ctrl = None
        self.current_obj_wxyz = self.obj_init_wxyz

        # GUI + 控件
        self._setup_gui()

        # 初始绘制网格 + 重置物体
        self._rebuild_grid_and_paint_blue()

        self.run_loop()

    def _reload_ik_wrapper(self):
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

        # 范围 + 分辨率控件
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

        # IK 参数控件
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

        self.btn_reload = self.server.gui.add_button("Reset")
        self.btn_solve = self.server.gui.add_button("Solve IK")

        self.txt = self.server.gui.add_text("Success", initial_value="0 / 0 (0.0%)")

        # 显示模式选择下拉
        self.combo_mode = self.server.gui.add_dropdown("mode",
                                                       options=["all", "success", "fail"],
                                                       initial_value="all")
        self.point_shape = self.server.gui.add_dropdown("point_shape",
                                                        options=["square", "diamond", "circle", "rounded", "sparkle"],
                                                        initial_value="circle")
        self.point_size = self.server.gui.add_slider("point_size", min=0.01, max=0.2, step=0.005, initial_value=0.1)
        # 播放模式 checkbox
        self.playing = self.server.gui.add_checkbox("playing", initial_value=False)

        # transform 控件 —用于目标物体交互
        self.obj_ctrl = self.server.scene.add_transform_controls(
            "/target_obj_ctrl",
            position=self.obj_init_pos,
            scale=0.5,
            wxyz=self.obj_init_wxyz
        )
        self.solver_state = 'idle'  # 'idle' / 'solving' / 'done'

        # 控件回调
        # 控制器 on_update：当用户拖 / 旋转物体时触发
        @self.obj_ctrl.on_update
        def _(evt, self=self):
            pos = tuple(self.obj_ctrl.position)
            wxyz = tuple(self.obj_ctrl.wxyz)
            if wxyz != self.current_obj_wxyz:
                self.current_obj_wxyz = wxyz
            else:
                return
            # 记录当前物体 pose
            self.target_obj_pose = (pos, wxyz)
            self._rebuild_grid_and_paint_blue()
            # 如果你想拖动物体后立即重新 solve，可以在这里调用
            # self._solve_async()

        # 防抖重建网格
        self._debounce_rebuild = Debouncer(0.20, self._rebuild_grid_and_paint_blue)
        for handle in [self.ms_x, self.ms_y, self.ms_z, self.sl_res]:
            @handle.on_update
            def _(_evt, self=self):
                self._debounce_rebuild.call()

        @self.btn_reload.on_click
        def _(_evt):
            # 从 GUI 读值更新 ik_params
            # self.ik_params["pos_tol"] = float(self.sl_pos.value)
            # self.ik_params["rot_tol"] = float(self.sl_rot.value)
            # self.ik_params["num_seeds"] = int(self.sl_seeds.value)
            # self.ik_params["self_collision_check"] = bool(self.cb_selfcol.value)
            # self.ik_params["self_collision_opt"] = bool(self.cb_selfopt.value)
            # self.ik_params["use_cuda_graph"] = bool(self.cb_cuda.value)
            # self._reload_ik_wrapper()
            # print("Reloaded IK params:", self.ik_params)
            self.sl_pos.value = self.ik_params["pos_tol"]
            self.sl_rot.value = self.ik_params["rot_tol"]
            self.sl_seeds.value = self.ik_params["num_seeds"]
            self.cb_selfcol.value = self.ik_params["self_collision_check"]
            self.cb_selfopt.value = self.ik_params["self_collision_opt"]
            self.cb_cuda.value = self.ik_params["use_cuda_graph"]
            # 重置物体位置
            self.obj_ctrl.position = self.obj_init_pos
            self.obj_ctrl.wxyz = self.obj_init_wxyz
            self.current_obj_wxyz = self.obj_init_wxyz
            self.target_obj_pose = (self.obj_init_pos, self.obj_init_wxyz)
            # 重建网格
            self._rebuild_grid_and_paint_blue()
            self._reload_ik_wrapper()

        @self.btn_solve.on_click
        def _(_evt):
            self._solve_async()

        @self.combo_mode.on_update
        def _(evt, self=self):
            self._apply_mode_filter()

        @self.playing.on_update
        def _(evt, self=self):
            if self.playing.value:
                self._playing()
        
        @self.point_shape.on_update
        def _(evt, self=self):
            if self.pc_handle is not None:
                self.pc_handle.point_shape = self.point_shape.value
            # self._rebuild_grid_and_paint_blue()
        
        @self.point_size.on_update
        def _(evt, self=self):
            if self.pc_handle is not None:
                self.pc_handle.point_size = self.point_size.value * self.sl_res.value
            # self._rebuild_grid_and_paint_blue()

    def _rebuild_grid_and_paint_blue(self):
        # 单位四元数
        # qu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # get qu from current obj_ctrl
        qu = torch.tensor(self.obj_ctrl.wxyz, dtype=torch.float32)
        

        x0, x1 = self.ms_x.value
        y0, y1 = self.ms_y.value
        z0, z1 = self.ms_z.value
        res = float(self.sl_res.value)

        self.grid_poses = create_grid_of_poses(x0, x1, y0, y1, z0, z1, res, qu)

        pts = self.grid_poses.position.cpu().numpy()

        # 初始都涂成蓝色
        colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
        colors[:, :] = np.array([0, 0, 255], dtype=np.uint8)

        size = self.sl_res.value * self.point_size.value
        
        if self.pc_handle is None:
            self.pc_handle = self.server.scene.add_point_cloud(
                name="/reach_pts", points=pts, colors=colors, point_size=size, point_shape=self.point_shape.value
            )
        else:
            self.pc_handle.points = pts
            self.pc_handle.colors = colors
            self.pc_handle.point_size = size
            self.pc_handle.point_shape = self.point_shape.value

        # 重置点状态
        self.point_state = np.full(pts.shape[0], -1, dtype=np.int8)
        self.point_sol = None

        self.target_obj_pose = (self.obj_ctrl.position, self.obj_ctrl.wxyz)

        self.txt.value = f"0 / {pts.shape[0]} (0.0%)"

    def _solve_async(self):
        if self.grid_poses is None:
            return

        def task():

            # 对整个网格求 IK
            self.txt.value = "Solving..."
            self.solver_state = 'solving'
            success, sols = self.ik_wrapper.solve(self.grid_poses)
            success = success.reshape(-1)

            pts = self.grid_poses.position.cpu().numpy()

            # 更新颜色数组（但暂不设置给 pc_handle，因为 mode 可能过滤）
            colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
            colors[success, :] = np.array([0, 255, 0], dtype=np.uint8)
            colors[~success, :] = np.array([255, 0, 0], dtype=np.uint8)

            # 更新状态 & 解
            self.point_state = np.zeros(pts.shape[0], dtype=np.int8)
            self.point_state[success] = 1
            self.point_state[~success] = 0
            self.point_sol = sols

            total = len(success)
            count = int(success.sum())
            rate = (count / total * 100.0) if total > 0 else 0.0
            self.txt.value = f"{count} / {total} ({rate:.1f}%)"
            self.solver_state = 'idle'

            # 最后根据当前 mode 过滤 + 上色
            self._apply_mode_filter()
        if self.solver_state == 'solving':
            print("Already solving, please wait.")
            self.txt.value = "Already solving, please wait."
            return
        th = threading.Thread(target=task, daemon=True)
        th.start()

    def _apply_mode_filter(self):
        if self.grid_poses is None or self.pc_handle is None or self.point_state is None:
            return

        mode = self.combo_mode.value
        n = self.point_state.shape[0]

        # 生成完整 pts + colors
        pts = self.grid_poses.position.cpu().numpy()
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[self.point_state == -1, :] = np.array([0, 0, 255], dtype=np.uint8)
        colors[self.point_state == 1, :] = np.array([0, 255, 0], dtype=np.uint8)
        colors[self.point_state == 0, :] = np.array([255, 0, 0], dtype=np.uint8)

        # 根据 mode 过滤 mask
        if mode == "all":
            mask = np.ones(n, dtype=bool)
        elif mode == "success":
            mask = (self.point_state == 1)
        elif mode == "fail":
            mask = (self.point_state == 0)
        else:
            mask = np.ones(n, dtype=bool)

        pts_f = pts[mask]
        colors_f = colors[mask]

        # 更新点云句柄
        self.pc_handle.points = pts_f
        self.pc_handle.colors = colors_f

    def _playing(self):
        def task():
            success_idxs = np.nonzero(self.point_state == 1)[0]
            if len(success_idxs) == 0:
                return
            for idx in success_idxs:
                if not self.playing.value:
                    break
                sol = self.point_sol[idx]
                sol_tensor = torch.tensor(sol, dtype=torch.float32)
                self.robot.set_joint_state(sol_tensor)
                time.sleep(0.1)
        th = threading.Thread(target=task, daemon=True)
        th.start()

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
    PointIKApp(args.robot_file)


if __name__ == "__main__":
    main()
