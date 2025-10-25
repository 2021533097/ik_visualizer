import os
import time
import threading

import torch
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

from curobo.util_file import load_yaml, join_path, get_robot_configs_path, get_assets_path
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig  # for kinematics & state

class FrankaVisualizer:
    def __init__(self, robot_cfg_file: str, urdf_path: str):
        # 初始化 Viser
        self.server = viser.ViserServer()

        # 加载 URDF 可视化对象
        # 读取 cuRobo robot config YAML
        cfg_all = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        kin = cfg_all["robot_cfg"]["kinematics"]
        urdf_rel = kin["urdf_path"]  # 可能是相对路径
        print(get_assets_path())
        print(kin['asset_root_path'])
        print(urdf_rel)
        # 构造完整路径（假设相对 cuRobo 的 robot configs 目录或 asset 目录）
        urdf_root = join_path(get_assets_path(), kin['asset_root_path'])
        urdf_full_path = join_path(get_assets_path(), urdf_rel)
        mesh_dir = join_path(urdf_root, "meshes")
        if not os.path.exists(urdf_full_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_full_path}")
        print(f"Loading URDF from: {urdf_full_path}\nMesh Dir: {mesh_dir}")
        urdf_obj = URDF.load(urdf_full_path, mesh_dir=mesh_dir, build_collision_scene_graph=True, load_collision_meshes=True)
        print(urdf_obj.scene, urdf_obj.collision_scene)
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf_obj,
            load_meshes=True,
            load_collision_meshes=True,
        )

        # 加载 cuRobo 的 robot 配置
        cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg_file))
        self.robot_cfg = RobotConfig.from_dict(cfg["robot_cfg"])
        self.tensor_args = TensorDeviceType()

        # 用 RobotWorld / RobotWorldConfig 或类似模块来做正运动学 / joint state
        # 这是 cuRobo 官方示例里常用的方法来得到 robot 的 kinematics / state。:contentReference[oaicite:0]{index=0}
        robot_world_cfg = RobotWorldConfig.load_from_config(robot_cfg_file, world_model={})
        self.robot_world = RobotWorld(robot_world_cfg)

        # 为可动关节创建滑条
        self.joint_sliders = []
        limits = self.viser_urdf.get_actuated_joint_limits()
        for jn, (low, high) in limits.items():
            low = low if low is not None else -3.14
            high = high if high is not None else 3.14
            init = (low + high) / 2.0
            sl = self.server.gui.add_slider(jn, min=low, max=high, step=1e-3, initial_value=init)
            self.joint_sliders.append((jn, sl))

        # 绑定滑条变化 -> 更新机械臂显示
        def on_any_joint_change(_evt):
            # 从所有滑条读值
            q = [sl.value for (_, sl) in self.joint_sliders]
            # 转 numpy 或 torch
            q_t = torch.tensor([q], **self.tensor_args.as_torch_dict())
            # 用 robot_world 得到 kinematics / link states
            state = self.robot_world.get_kinematics(q_t)
            # state 包含 link positions / quaternions
            link_pos = state.links_position[0].detach().cpu().numpy()
            link_quat = state.links_quaternion[0].detach().cpu().numpy()
            # 更新 viser_urdf 显示
            self.viser_urdf.update_cfg(q)

        for (_, sl) in self.joint_sliders:
            @sl.on_update
            def _(evt, self=self):
                on_any_joint_change(evt)

        # 初始化一次显示
        init_q = [sl.value for (_, sl) in self.joint_sliders]
        self.viser_urdf.update_cfg(init_q)

        # 主循环保持
        while True:
            time.sleep(0.1)


def main():
    # 环境设置，例如指定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 请你传入你的配置文件名与 urdf 路径
    # 假设 franka.yml 是 robot config，franka.urdf 是 URDF 文件
    visual = FrankaVisualizer(robot_cfg_file="franka.yml", urdf_path="franka.urdf")


if __name__ == "__main__":
    main()
