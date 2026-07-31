"""测试 SnapMoGen GT motion 可视化。

读取原始 GT motion npy 文件，调用 visualize_snapmogen 渲染为 MP4，
同时计算并打印各骨骼段的长度，用于检查"手特别小臂特别长"的渲染问题。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.visualize.vis_snapmogen import (
    visualize_snapmogen,
    recover_joints_from_features,
    SNAPMOGEN_KINEMATIC_CHAIN,
)

# ── 关节名称映射（根据 SNAPMOGEN_KINEMATIC_CHAIN 推测） ──
# Chain: [0, 1, 2, 3, 4, 5, 6]   → spine: 0=pelvis, 1=spine1, 2=spine2, 3=chest, 4=neck, 5,6=head
# Chain: [3, 7, 8, 9, 10]         → left arm: 3=chest, 7=L_shoulder, 8=L_elbow, 9=L_wrist, 10=L_hand
# Chain: [3, 11, 12, 13, 14]      → right arm: 3=chest, 11=R_shoulder, 12=R_elbow, 13=R_wrist, 14=R_hand
# Chain: [0, 15, 16, 17, 18, 19]  → left leg: 0=pelvis, 15=L_hip, 16=L_knee, 17=L_ankle, 18=L_foot, 19=L_toe
# Chain: [15, 20, 21, 22, 23]     → right leg: 15=L_hip, 20=R_hip, 21=R_knee, 22=R_ankle, 23=R_foot
JOINT_NAMES = {
    0: "Pelvis",
    1: "Spine1",
    2: "Spine2",
    3: "Chest",
    4: "Neck",
    5: "Head1",
    6: "Head2",
    7: "L_Shoulder",
    8: "L_Elbow",
    9: "L_Wrist",
    10: "L_Hand",
    11: "R_Shoulder",
    12: "R_Elbow",
    13: "R_Wrist",
    14: "R_Hand",
    15: "L_Hip",
    16: "L_Knee",
    17: "L_Ankle",
    18: "L_Foot",
    19: "L_Toe",
    20: "R_Hip",
    21: "R_Knee",
    22: "R_Ankle",
    23: "R_Foot",
}

# 四肢段定义：(起点, 终点, 描述)
LIMB_SEGMENTS = [
    (3, 7, "UpperArm_L (Chest→L_Shoulder)"),
    (7, 8, "Forearm_L (L_Shoulder→L_Elbow)"),
    (8, 9, "Hand_L (L_Elbow→L_Wrist)"),
    (9, 10, "Finger_L (L_Wrist→L_Hand)"),
    (3, 11, "UpperArm_R (Chest→R_Shoulder)"),
    (11, 12, "Forearm_R (R_Shoulder→R_Elbow)"),
    (12, 13, "Hand_R (R_Elbow→R_Wrist)"),
    (13, 14, "Finger_R (R_Wrist→R_Hand)"),
    (0, 15, "Thigh_L (Pelvis→L_Hip)"),
    (15, 16, "Shin_L (L_Hip→L_Knee)"),
    (16, 17, "Foot_L (L_Knee→L_Ankle)"),
    (17, 18, "Toe_L (L_Ankle→L_Foot)"),
    (15, 20, "Thigh_R (L_Hip→R_Hip)"),  # 注意：右腿链从 L_Hip 开始
    (20, 21, "Shin_R (R_Hip→R_Knee)"),
    (21, 22, "Foot_R (R_Knee→R_Ankle)"),
    (22, 23, "Toe_R (R_Ankle→R_Foot)"),
    (0, 1, "Spine (Pelvis→Spine1)"),
    (1, 2, "Spine (Spine1→Spine2)"),
    (2, 3, "Spine (Spine2→Chest)"),
    (3, 4, "Neck (Chest→Neck)"),
]


def compute_segment_lengths(joints):
    """计算各骨骼段的平均长度（所有帧的平均值）。

    Args:
        joints: (T, 24, 3) 全局关节坐标

    Returns:
        dict: {描述: 平均长度}
    """
    lengths = {}
    for j1, j2, desc in LIMB_SEGMENTS:
        diff = joints[:, j1] - joints[:, j2]  # (T, 3)
        dist = np.linalg.norm(diff, axis=1)    # (T,)
        lengths[desc] = {
            'mean': float(dist.mean()),
            'std': float(dist.std()),
            'min': float(dist.min()),
            'max': float(dist.max()),
        }
    return lengths


def main():
    # ── 配置 ──
    motion_path = 'dataset/SnapMoGen/renamed_feats/dd_00000.npy'
    output_dir = 'output/test_vis_gt'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dd_00000_gt.mp4')

    # ── 加载 GT motion ──
    print(f'加载 GT motion: {motion_path}')
    motion = np.load(motion_path)
    print(f'  shape: {motion.shape}')
    print(f'  dtype: {motion.dtype}')
    print(f'  range: [{motion.min():.4f}, {motion.max():.4f}]')

    # ── 打印各维度的数值范围 ──
    print(f'\n各维度统计:')
    print(f'  dim 0 (root ang vel):          mean={motion[:, 0].mean():.6f}, std={motion[:, 0].std():.6f}')
    print(f'  dim 1-2 (root lin vel XZ):     mean={motion[:, 1:3].mean():.6f}, std={motion[:, 1:3].std():.6f}')
    print(f'  dim 3 (root height Y):         mean={motion[:, 3].mean():.4f}, std={motion[:, 3].std():.4f}')
    print(f'  dim 4-147 (rotations 24×6):    mean={motion[:, 4:148].mean():.6f}, std={motion[:, 4:148].std():.6f}')
    print(f'  dim 148-219 (positions 24×3):  mean={motion[:, 148:220].mean():.4f}, std={motion[:, 148:220].std():.4f}')
    print(f'  dim 220-291 (velocities 24×3): mean={motion[:, 220:292].mean():.6f}, std={motion[:, 220:292].std():.6f}')
    print(f'  dim 292-295 (foot contact):    mean={motion[:, 292:296].mean():.6f}, std={motion[:, 292:296].std():.6f}')

    # ── 恢复关节并检查骨骼长度 ──
    print(f'\n恢复关节坐标...')
    # 直接使用原始特征（无需反归一化，因为这是 GT 原始空间数据）
    joints = recover_joints_from_features(motion, joints_num=24)
    print(f'  joints shape: {joints.shape}')
    print(f'  joints range: X=[{joints[:, :, 0].min():.4f}, {joints[:, :, 0].max():.4f}], '
          f'Y=[{joints[:, :, 1].min():.4f}, {joints[:, :, 1].max():.4f}], '
          f'Z=[{joints[:, :, 2].min():.4f}, {joints[:, :, 2].max():.4f}]')

    # 计算各骨骼段长度
    print(f'\n骨骼段长度分析 (所有帧平均):')
    seg_lengths = compute_segment_lengths(joints)

    # 分组打印
    for group_name, segments in [
        ("左臂 (Left Arm)", [
            "UpperArm_L (Chest→L_Shoulder)",
            "Forearm_L (L_Shoulder→L_Elbow)",
            "Hand_L (L_Elbow→L_Wrist)",
            "Finger_L (L_Wrist→L_Hand)",
        ]),
        ("右臂 (Right Arm)", [
            "UpperArm_R (Chest→R_Shoulder)",
            "Forearm_R (R_Shoulder→R_Elbow)",
            "Hand_R (R_Elbow→R_Wrist)",
            "Finger_R (R_Wrist→R_Hand)",
        ]),
        ("左腿 (Left Leg)", [
            "Thigh_L (Pelvis→L_Hip)",
            "Shin_L (L_Hip→L_Knee)",
            "Foot_L (L_Knee→L_Ankle)",
            "Toe_L (L_Ankle→L_Foot)",
        ]),
        ("右腿 (Right Leg)", [
            "Thigh_R (L_Hip→R_Hip)",
            "Shin_R (R_Hip→R_Knee)",
            "Foot_R (R_Knee→R_Ankle)",
            "Toe_R (R_Ankle→R_Foot)",
        ]),
    ]:
        print(f'\n  【{group_name}】')
        for seg_name in segments:
            if seg_name in seg_lengths:
                s = seg_lengths[seg_name]
                print(f'    {seg_name:<45s}: mean={s["mean"]:.4f}, std={s["std"]:.4f}, range=[{s["min"]:.4f}, {s["max"]:.4f}]')

    # ── 渲染可视化 ──
    print(f'\n开始渲染 MP4...')
    result = visualize_snapmogen(
        motion=motion,
        output_path=output_path,
        caption='dd_00000 GT (original features, no denorm)',
        mean=None,
        std=None,
        fps=30,
    )

    print(f'\n渲染完成: {result}')
    file_size_mb = os.path.getsize(result) / (1024 * 1024)
    print(f'文件大小: {file_size_mb:.2f} MB')
    print(f'\n请检查 {output_path} 查看渲染结果是否正常。')


if __name__ == '__main__':
    main()
