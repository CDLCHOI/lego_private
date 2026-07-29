"""SnapMoGen motion 可视化工具。

将 SnapMoGen 的 296 维 motion 特征恢复为 24 关节的全局坐标，
并渲染为 MP4 视频文件。

注意：本模块所有函数均从 SnapMoGen_zhiwei 工程迁移而来，
不进行任何外部软链接导入。
"""

import math
import os
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation

from utils.quaternion import qrot, qinv

# ==============================================================================
# SnapMoGen 24 关节 kinematic chain（从 SnapMoGen_zhiwei/utils/paramUtil.py 迁移）
# ==============================================================================
SNAPMOGEN_KINEMATIC_CHAIN = [
    [0, 1, 2, 3, 4, 5, 6],
    [3, 7, 8, 9, 10],
    [3, 11, 12, 13, 14],
    [0, 15, 16, 17, 18, 19],
    [15, 20, 21, 22, 23],
]


# ==============================================================================
# 关节恢复函数（从 SnapMoGen_zhiwei/utils/motion_process_bvh.py 迁移）
# ==============================================================================
def recover_root_rot_pos(data: torch.Tensor):
    """从 SnapMoGen 296 维特征中恢复根节点旋转四元数和世界坐标位置。

    输入:
        data: (..., 296) 的 SnapMoGen 特征
              [..., 0]    = 根节点旋转角速度
              [..., 1:3]  = 根节点线速度 (xz)
              [..., 3]    = 根节点高度 (y)

    返回:
        r_rot_quat: (..., 4) 根节点旋转四元数
        r_pos:      (..., 3) 根节点世界坐标位置
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # 从旋转角速度获取 Y 轴旋转角度
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang / 2, dim=-1)

    r_rot_quat = data.new_zeros(data.shape[:-1] + (4,))
    # (cos(r/2), 0, sin(r/2), 0)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # 获取根节点位置
    r_pos = data.new_zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    # 对根节点位置施加 Y 轴旋转
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_joints_from_features(features: Union[torch.Tensor, np.ndarray],
                                 joints_num: int = 24) -> np.ndarray:
    """从 SnapMoGen 296 维特征恢复 24 关节的全局坐标。

    输入:
        features:   (T, 296) 的 SnapMoGen 原始特征（已反归一化到原始空间）
        joints_num: 关节数量，SnapMoGen 固定为 24

    返回:
        joints: (T, 24, 3) 的全局关节坐标 numpy 数组
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))

    # 确保输入是 2D
    if features.ndim == 2:
        features = features.unsqueeze(0)  # (1, T, 296)

    r_rot_quat, r_pos = recover_root_rot_pos(features)

    # RIC 位置索引: 1(root_ang_vel) + 2(root_lin_vel) + 1(root_y) + joints_num*6(rotations)
    start_indx = 1 + 2 + 1 + joints_num * 6  # = 148
    end_indx = start_indx + joints_num * 3      # = 220

    positions = features[..., start_indx:end_indx]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # 对局部关节施加 Y 轴旋转
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )

    # 对关节施加根节点 XZ
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    return positions.squeeze(0).cpu().numpy()  # (T, 24, 3)


# ==============================================================================
# 渲染辅助类型和函数
# ==============================================================================
JointArray = Union[np.ndarray, torch.Tensor]
OutputPath = Union[str, Path]


def _as_float32_numpy(joints: JointArray) -> np.ndarray:
    """验证关节数据并返回连续的 CPU float32 数组。"""
    if isinstance(joints, torch.Tensor):
        if joints.device.type != "cpu":
            raise ValueError(
                "joints 必须是 CPU Tensor；got device {}".format(joints.device)
            )
        if joints.layout != torch.strided:
            raise TypeError(
                "joints 必须是 dense CPU Tensor；got layout {}".format(joints.layout)
            )
        shape = tuple(joints.shape)
    elif isinstance(joints, np.ndarray):
        shape = joints.shape
    else:
        raise TypeError("joints 必须是 NumPy 数组或 CPU Torch tensor")

    if len(shape) != 3 or shape[1:] != (24, 3):
        raise ValueError(
            "joints 必须有精确的形状 [T, 24, 3]；got {}".format(shape)
        )
    if shape[0] == 0:
        raise ValueError("joints 必须包含至少一帧")

    if isinstance(joints, torch.Tensor):
        if joints.is_complex():
            raise TypeError("不支持复数关节坐标")
        array = joints.detach().to(dtype=torch.float32).numpy()
    else:
        if np.iscomplexobj(joints):
            raise TypeError("不支持复数关节坐标")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(joints, dtype=np.float32)

    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("joints 必须只包含有限坐标")
    return array


def _positive_fps(fps: float) -> float:
    """验证并返回正数 FPS。"""
    try:
        value = float(fps)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("fps 必须是正有限数") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps 必须是正有限数")
    return value


def _animation_data(joints: np.ndarray):
    """准备落地、水平根节点相对坐标的动画数据。"""
    grounded = joints.astype(np.float64, copy=True)
    grounded[..., 1] -= grounded[..., 1].min()

    root_trajectory = grounded[:, 0][:, [0, 2]].copy()
    root_relative = grounded.copy()
    root_relative[..., 0] -= grounded[:, 0:1, 0]
    root_relative[..., 2] -= grounded[:, 0:1, 2]

    skeleton_extent = float(
        np.abs(root_relative[..., [0, 2]]).max(initial=0.0)
    )
    trajectory_extent = float(np.ptp(root_trajectory, axis=0).max(initial=0.0))
    horizontal_radius = max(0.75, skeleton_extent, trajectory_extent) * 1.15
    vertical_limit = max(1.0, float(grounded[..., 1].max()) * 1.10)
    return root_relative, root_trajectory, horizontal_radius, vertical_limit


def render_motion_mp4(
    joints: JointArray,
    output_path: OutputPath,
    caption: str = "",
    *,
    fps: float = 30,
) -> Path:
    """将 ``[T, 24, 3]`` 世界空间关节数据渲染为新的 MP4 文件。

    输入:
        joints:      (T, 24, 3) 的关节坐标
        output_path: 输出 MP4 文件路径
        caption:     视频标题（可选）
        fps:         帧率，默认 30

    返回:
        已创建的输出文件 Path
    """
    data = _as_float32_numpy(joints)
    frames_per_second = _positive_fps(fps)
    try:
        destination = Path(output_path).expanduser()
    except TypeError as error:
        raise TypeError("output_path 必须是字符串或 path-like 值") from error

    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite existing output: {}".format(destination)
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "渲染 MP4 需要 ffmpeg，但系统中未找到 ffmpeg"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    root_relative, root_trajectory, radius, vertical_limit = _animation_data(data)
    wrapped_caption = textwrap.fill(str(caption), width=60)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            suffix=".tmp.mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        figure = None
        try:
            figure = plt.figure(figsize=(8, 8))
            axes = figure.add_subplot(111, projection="3d")
            figure.suptitle(wrapped_caption, fontsize=14, y=0.96)
            figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.91)

            plane_x, plane_horizontal_z = np.meshgrid(
                np.array([-radius, radius], dtype=np.float64),
                np.array([-radius, radius], dtype=np.float64),
            )
            plane_height = np.zeros_like(plane_x)
            chain_colors = (
                "#222222",
                "#d1495b",
                "#3976af",
                "#d1495b",
                "#3976af",
            )

            def update(frame_index):
                axes.cla()
                axes.set_xlim3d(-radius, radius)
                axes.set_ylim3d(-radius, radius)
                axes.set_zlim3d(0.0, vertical_limit)
                axes.set_box_aspect(
                    (2.0 * radius, 2.0 * radius, vertical_limit)
                )
                axes.view_init(elev=18, azim=-65)
                axes.set_axis_off()

                axes.plot_surface(
                    plane_x,
                    plane_horizontal_z,
                    plane_height,
                    color="#b8b8b8",
                    alpha=0.22,
                    linewidth=0,
                    shade=False,
                )

                current_root = root_trajectory[frame_index]
                trajectory = root_trajectory[: frame_index + 1] - current_root
                axes.plot3D(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    np.zeros(frame_index + 1, dtype=np.float64),
                    color="#4169e1",
                    linewidth=1.8,
                )

                pose = root_relative[frame_index]
                for chain, color in zip(SNAPMOGEN_KINEMATIC_CHAIN, chain_colors):
                    axes.plot3D(
                        pose[chain, 0],
                        pose[chain, 2],
                        pose[chain, 1],
                        color=color,
                        linewidth=4.0,
                        solid_capstyle="round",
                    )

            animation = FuncAnimation(
                figure,
                update,
                frames=data.shape[0],
                interval=1000.0 / frames_per_second,
                repeat=False,
                blit=False,
            )
            writer = FFMpegWriter(
                fps=frames_per_second,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
            animation.save(str(temporary_path), writer=writer)
        except Exception as error:
            raise RuntimeError(
                "ffmpeg 编码 motion 到 {} 时失败: {}".format(destination, error)
            ) from error
        finally:
            if figure is not None:
                plt.close(figure)

        try:
            os.link(str(temporary_path), str(destination))
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite output created while rendering: {}".format(
                    destination
                )
            ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return destination


# ==============================================================================
# 主可视化入口函数
# ==============================================================================
def visualize_snapmogen(
    motion: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    caption: str = "",
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    fps: float = 30,
    m_length: Optional[int] = None,
) -> Path:
    """将 SnapMoGen motion 特征可视化并保存为 MP4 文件。

    输入:
        motion:      (T, 296) 的 SnapMoGen 特征。
                     如果 mean/std 不为 None，表示特征仍处于归一化空间，
                     函数内部会先反归一化再恢复关节。
                     如果 mean/std 为 None，则假定特征已在原始空间。
        output_path: 输出 MP4 文件路径，如 ``/path/to/vis/sample_0.mp4``
        caption:     视频标题文本（可选）
        mean:        数据集均值 (296,)，用于反归一化。为 None 则跳过。
        std:         数据集标准差 (296,)，用于反归一化。为 None 则跳过。
        fps:         帧率，默认 30
        m_length:    有效运动长度（帧数），超出部分将被裁剪。
                     为 None 则使用全部帧。

    返回:
        已创建的 MP4 文件 Path

    示例::

        motion = np.load('dataset/SnapMoGen/renamed_feats/dd_00000.npy')
        visualize_snapmogen(motion, 'output/vis/test.mp4', caption='test')
    """
    if isinstance(motion, torch.Tensor):
        motion_np = motion.detach().cpu().numpy()
    else:
        motion_np = np.asarray(motion, dtype=np.float32)

    # 确保是 2D
    if motion_np.ndim == 1:
        motion_np = motion_np.reshape(1, -1)

    # 反归一化
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
        motion_np = motion_np * std.astype(np.float32) + mean.astype(np.float32)

    # 裁剪到有效长度
    if m_length is not None and m_length < motion_np.shape[0]:
        motion_np = motion_np[:m_length]

    # 过滤全零填充帧
    nonzero_mask = np.any(motion_np != 0, axis=1)
    if not nonzero_mask.all():
        valid_len = nonzero_mask.sum()
        if valid_len > 0:
            motion_np = motion_np[:valid_len]

    # 恢复关节
    joints = recover_joints_from_features(motion_np, joints_num=24)

    # 渲染并保存 MP4
    output_file = render_motion_mp4(joints, output_path, caption=caption, fps=fps)
    print(f'[SnapMoGen Vis] 已保存可视化视频: {output_file}')
    return output_file


__all__ = [
    "visualize_snapmogen",
    "render_motion_mp4",
    "recover_joints_from_features",
    "recover_root_rot_pos",
    "SNAPMOGEN_KINEMATIC_CHAIN",
]
