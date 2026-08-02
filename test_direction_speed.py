"""
测试生成的 motion 是否真正遵循 motion-related keyword（rebuttal 纯文本版）

方案详见 test_direction_speed.md。四类关键词统一汇报为「每秒的 XX」标量：
    left/right          : 每秒横向位移      (m/s,  + = 向左)      固定第0帧参考系
    slowly/quickly      : 每秒位移大小      (m/s,  恒正)
    forward/backward    : 每秒沿自身朝向位移 (m/s,  + = 前进)      逐帧当前朝向
    clockwise/ccw       : 每秒朝向转角      (deg/s,+ = 逆时针)    逐帧当前朝向

用法:
    python test_direction_speed.py --calibrate     # 只跑 GT 符号校准, 不需要 GPU/模型
    python test_direction_speed.py                 # 完整实验
"""
import os
import sys
import csv
import json
import hashlib

import numpy as np
import torch

# --calibrate 不走 option_transformer, 需先摘掉再解析
CALIBRATE_ONLY = '--calibrate' in sys.argv
if CALIBRATE_ONLY:
    sys.argv.remove('--calibrate')

import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('DS_GPU', '1')   # DS_GPU=1 可换卡

from utils.motion_process import recover_from_ric


# ==================== 配置 ====================
FPS = 20
GEN_LENGTH = 196                 # 固定生成长度(帧), 9.8 秒
N_SAMPLES = 20                   # 每条文本的样本数
BASE_SEED = 20260730
DIFFUSION_STEPS = 50

# (窗口名, 起始帧, 结束帧)
EVAL_WINDOWS = [('3s', 0, 60), ('6s', 0, 120), ('full', 0, 196)]

# HumanML3D 关节索引: face_joint_indx = [2,1,17,16] -> r_hip, l_hip, sdr_r, sdr_l
R_HIP, L_HIP, R_SDR, L_SDR = 2, 1, 17, 16
L_WRIST, R_WRIST = 20, 21         # SMPL 22 关节序: 20=左手腕, 21=右手腕(已用 GT 校准 17/20)
FOOT_IDS = [10, 11]              # 与 utils/metrics.py:489 的 skating ratio 保持一致
LEG_JOINTS = [10, 11]             # 左右脚, 用于计算腿部相对运动速度（跑/走的核心差异在末端）
UP = np.array([0.0, 1.0, 0.0])

SMOOTH_W = 5                     # 朝向向量的滑动平均窗口
YAW_CLIP_DEG = 30.0              # 单帧转角上限(20fps 下 = 600 deg/s), 超出视为异常跳变
FOOT_HEIGHT_THRESH = 0.05        # 触地判定: 脚高 < 5cm
MOVE_EPS = 0.1                   # 有效运动的速度阈值 (m/s)
NO_MOVE_PATH = 0.3               # 窗口内路径长度 < 0.3m 视为静止样本

MEAN_PATH = './dataset/HumanML3D/Mean.npy'
STD_PATH = './dataset/HumanML3D/Std.npy'
GT_JOINT_DIR = 'dataset/HumanML3D/new_joints'
GT_TEXT_DIR = 'dataset/HumanML3D/texts'

CACHE_DIR = 'output/direction_speed_cache'
OUT_SAMPLES_CSV = 'test_direction_speed_samples.csv'
OUT_SUMMARY_CSV = 'test_direction_speed_summary.csv'

MODELS = [
    # (显示名, ckpt 路径, 是否 lora)
    ('MDM', 'output/0814_MDMCLIP_b128/net_best.pth', False),
    ('LeGO', 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ricglobal1/net_best.pth', True),
    ('LeGO-0', 'output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/net_best.pth', True),
    ('MDM+LeGO-CLIP', 'output/0911_MDMCLIP_preatrainlora_ric1_b64/net_best.pth', True),
]

# 每类: 主判据 key, 单位, A/B 语义标签, 汇报窗口, 5 组文本对(A = 期望主判据更大的一侧)
CATEGORIES = [
    {
        'name': 'left_right',
        'metric': 'disp_angle',
        'unit': 'deg',
        'angle_key': None,
        'angle_desc': None,
        'constraint': None,
        'label_a': 'left', 'label_b': 'right',
        'signed': True,
        'filter_no_move': True,          # 角度除以位移模长, 静止样本是纯噪声, 必须剔除
        'report_window': '6s',       # 净位移会被 U 型回转抵消, 用短窗口(见 md §4.1)
        'desc': 'direction of travel relative to the initial facing direction (degrees, +90 = straight left, -90 = straight right, 0 = straight forward)',
        'pairs': [
            ('A person walks toward left', 'A person walks toward right'),
            ('A person walks to the left', 'A person walks to the right'),
            ('A person turns left and keeps walking', 'A person turns right and keeps walking'),
            ('A man steps to his left side', 'A man steps to his right side'),
            ('A person is walking to the left direction', 'A person is walking to the right direction'),
        ],
    },
    {
        'name': 'slow_quick',
        'metric': 'm_speed',
        'unit': 'm/s',
        'angle_key': None,
        'angle_desc': None,
        'constraint': None,
        'label_a': 'quickly', 'label_b': 'slowly',
        'signed': False,          # 速度恒正, 无绝对符号可言
        'filter_no_move': False,  # 静止本身就是"慢"的信息, 剔除会扭曲这一类
        'report_window': 'full',
        'desc': 'travelled distance per second',
        'pairs': [
            ('A person walks quickly', 'A person walks slowly'),
            ('A person walks forward quickly', 'A person walks forward slowly'),
            ('A person is walking at a fast pace', 'A person is walking at a slow pace'),
            ('A man walks very quickly', 'A man walks very slowly'),
            ('A person quickly moves forward', 'A person slowly moves forward'),
        ],
    },
    {
        'name': 'fwd_bwd',
        'metric': 'm_align',
        'unit': 'cos',
        'angle_key': 'align_angle',      # 展示用: 走向与朝向夹角(度)
        'angle_desc': '走向与自身朝向的夹角, 值域0~180, 0=完全朝前走 90=纯侧移 180=完全倒着走',
        # 文本明确说了 "facing forward", 故额外要求朝向保持。
        # 使用逐帧朝向偏离均值(而非首末夹角): 弧形倒走时首末夹角可能 >90°,
        # 但逐帧均值 ≈ 总偏转/2, 不会误杀真正的倒走。转身作弊前几帧就跳到 >60°, 会被抓住。
        'constraint': {'key': 'mean_heading_dev', 'max': 60.0,
                       'desc': '逐帧朝向偏离初始方向均值 < 60度 (即文本要求的 "facing forward" 被保持, 允许缓慢弧形偏转)'},
        'label_a': 'forward', 'label_b': 'backward',
        'signed': True,
        'filter_no_move': True,          # 余弦要除以里程, 静止样本是纯噪声
        'report_window': 'full',     # 归一化后不受时长影响, 用全程
        'desc': "angle between the walking direction and the character's own facing direction "
                "(cosine, +1 = walking forward, -1 = walking backward)",
        # 5 对都显式锁定"面朝前", 否则 "walks backward" 有歧义(转身走过去也能算),
        # 等于用一个模型没被告知的约束去惩罚它。词频: facing forward 114, faces forward 14,
        # while facing 35, straight line 323 —— 均在训练集中存在(keeps facing 为 0, 已避开)。
        'pairs': [
            ('A person faces forward and walks forward',
             'A person faces forward and walks backward'),
            ('A person is facing forward and walks forward',
             'A person is facing forward and walks backward'),
            ('A person walks forward while facing forward',
             'A person walks backward while facing forward'),
            ('A person is facing forward and steps forward',
             'A person is facing forward and steps backward'),
            ('A person faces forward and walks forward in a straight line',
             'A person faces forward and walks backward in a straight line'),
        ],
    },
    {
        'name': 'cw_ccw',
        'metric': 'm_yaw',
        'unit': 'deg/s',
        'angle_key': None,
        'angle_desc': None,
        'constraint': None,
        'label_a': 'counterclockwise', 'label_b': 'clockwise',
        'signed': True,
        'filter_no_move': False,   # 原地转身是 "turns clockwise" 的合法实现, 不能按位移剔除
        'report_window': 'full',
        'desc': 'turning rate of the facing direction (positive = counterclockwise)',
        'pairs': [
            ('A person walks in a circle counterclockwise', 'A person walks in a circle clockwise'),
            ('A person turns counterclockwise', 'A person turns clockwise'),
            ('A person walks counterclockwise', 'A person walks clockwise'),
            ('A man rotates his body counterclockwise', 'A man rotates his body clockwise'),
            ('A person moves along a counter-clockwise path', 'A person moves along a clockwise path'),
        ],
    },
    {
        'name': 'hand_lr',
        'metric': 'm_hand',
        'unit': 'm',
        'angle_key': None,
        'angle_desc': None,
        'constraint': None,
        'label_a': 'left hand', 'label_b': 'right hand',
        'signed': True,
        'filter_no_move': False,   # 主判据是手腕高度, 与根节点位移无关, 不按位移剔除
        'report_window': 'full',
        'desc': 'height difference between the left and right wrist (positive = left wrist higher)',
        # 词频: raises their 1.32%, arm up 0.74%, while holding 0.61%, hand up 0.56%,
        # walks forward while 0.33%, with his left arm 0.20% —— 均为训练集高频短语;
        # 训练集中有 198 条 caption 同时含"走路 + 举手 + 左/右手", 概念覆盖充分。
        # (keeps walking 仅 29 条, 已避开)
        'pairs': [
            ('A person walks forward while raising their left arm',
             'A person walks forward while raising their right arm'),
            ('A person walks forward and raises their left hand up high',
             'A person walks forward and raises their right hand up high'),
            ('A person is walking with his left arm raised',
             'A person is walking with his right arm raised'),
            ('A person walks while holding their left hand up in the air',
             'A person walks while holding their right hand up in the air'),
            ('A man walks forward and lifts his left arm up',
             'A man walks forward and lifts his right arm up'),
        ],
    },
    {
        'name': 'walk_run',
        'metric': 'leg_speed',
        'unit': 'm/s',
        'angle_key': None,
        'angle_desc': None,
        'constraint': None,
        'label_a': 'run', 'label_b': 'walk',
        'signed': False,          # 速度恒正, 只比大小
        'filter_no_move': False,  # 原地跑步根节点不动但腿在高速运动, 不能按根位移剔除
        'report_window': 'full',
        'desc': 'foot speed relative to root (m/s), removes global translation to capture in-place running',
        'pairs': [
            ('A person runs forward', 'A person walks forward'),
            ('A person is running', 'A person is walking'),
            ('A man runs forward', 'A man walks forward'),
            ('A person runs in a straight line', 'A person walks in a straight line'),
            ('The person runs forward quickly', 'The person walks forward slowly'),
        ],
    },
]


# ==================== 通用几何底层 (md §3) ====================
def hproj(v):
    """投影到水平面(置零 y 分量)。v: (...,3)"""
    out = np.array(v, dtype=np.float64, copy=True)
    out[..., 1] = 0.0
    return out


def normalize_vec(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def moving_average(x, w):
    """沿时间轴(axis=0)滑动平均, 边界用 edge padding。x: (T,C)"""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
    ker = np.ones(w) / w
    return np.stack([np.convolve(xp[:, c], ker, mode='valid') for c in range(x.shape[1])], axis=1)


def body_frames(joints, smooth_w=SMOOTH_W):
    """逐帧身体坐标系。

    joints: (T,22,3) 世界坐标, y 为高度
    返回 right, forward, left, 均为 (T,3) 水平单位向量

    across 由 (r_hip - l_hip) + (r_sdr - l_sdr) 定义, 是解剖学上指向人体右侧的向量。
    实测在 HumanML3D 上 right 恒为 [-1,0,0], forward 恒为 [0,0,+1], 即 +X 是人体左侧。
    """
    across = (joints[:, R_HIP] - joints[:, L_HIP]) + (joints[:, R_SDR] - joints[:, L_SDR])
    across = moving_average(hproj(across), smooth_w)
    right = normalize_vec(across)
    forward = normalize_vec(np.cross(np.broadcast_to(UP, right.shape), right))
    return right, forward, -right


def root_horizontal(joints):
    """根节点水平位置 (T,3)"""
    return hproj(joints[:, 0])


def frame_disp(p, s, e):
    """窗口内逐帧位移 (N,3), N = e-s-1"""
    return p[s + 1:e] - p[s:e - 1]


def signed_yaw_increments(vecs, clip_deg=YAW_CLIP_DEG):
    """逐帧有符号转角增量(弧度)。vecs: (T,3) 水平单位向量 -> (T-1,)

    delta_t = atan2(<cross(v_t, v_t+1), up>, <v_t, v_t+1>)
    正号 = 绕 +Y 正旋转 = 把 +Z 转向 +X = 转向人体左侧 = 俯视逆时针。
    """
    v0, v1 = vecs[:-1], vecs[1:]
    cross_y = np.cross(v0, v1)[..., 1]        # <cross(v0,v1), up>
    dot = np.sum(v0 * v1, axis=-1)
    delta = np.arctan2(cross_y, dot)
    lim = np.deg2rad(clip_deg)
    return np.clip(delta, -lim, lim)


def foot_contact_from_joints(joints, thresh=FOOT_HEIGHT_THRESH):
    """触地判定 (T,2) bool, 与 skating ratio 用同一套脚部关节和高度阈值"""
    return joints[:, FOOT_IDS, 1] < thresh


def count_steps(contact, s, e):
    """窗口内触地上升沿总数(左右脚相加), 即步数"""
    c = contact[s:e]
    return int(np.logical_and(~c[:-1], c[1:]).sum())


# ==================== 四类主判据 (md §4) ====================
def metric_lateral(joints, s, e, cache):
    """left/right: 位移方向相对第 0 帧朝向偏了多少 —— 纯方向量, 不掺速度。+ = 向左

    主判据 m_lat_cos = <d, L0> / |d|, 分母是净位移模长, 把"走多远/多快"约掉。
    若用未归一化的 <d,L0>/dt (m/s), 则"走得远"会被误当成"方向更准", 那是 speed 类的事。
    """
    L0 = cache['left'][s]
    F0 = cache['forward'][s]
    d = cache['p'][e - 1] - cache['p'][s]
    dn = float(np.linalg.norm(d))
    dt = cache['dt']
    disp = cache['disp']
    return {
        # 主判据: 净位移方向在"左"轴上的余弦分量, [-1,1], +1 = 正左, -1 = 正右
        'm_lat_cos': float(np.dot(d, L0)) / dn if dn > 1e-6 else 0.0,
        # 展示用: 位移方向角(度), 0 = 正前, +90 = 正左, -90 = 正右
        'disp_angle': float(np.degrees(np.arctan2(np.dot(d, L0), np.dot(d, F0)))) if dn > 1e-6 else 0.0,
        # 辅助: 未归一化版本(m/s), 说明幅度
        'm_lat': float(np.dot(d, L0)) / dt,
        # 辅助: 用每帧当前左轴 -> 侧身平移(strafe)分量
        'v_lat_body': float(np.sum(disp * cache['left'][s:e - 1])) / dt,
    }


def metric_speed(joints, s, e, cache):
    """slowly/quickly / walk/run: 每秒位移大小(路径长度/移动时长), 恒正

    只统计根节点真正在移动的帧(速度 > MOVE_EPS m/s), 排除静止帧和原地转身帧。
    这避免"走走停停"的motion因为停的时间长而被误判为"慢"。
    """
    dt = cache['dt']
    step = np.linalg.norm(cache['disp'], axis=-1)
    moving = step * FPS > MOVE_EPS
    n_moving = max(moving.sum(), 1)
    # 主判据: 总路径 / 移动时长 (排除静止帧的稀释效应)
    m_speed = float(cache['path']) / (n_moving / FPS) if moving.any() else 0.0
    return {
        'm_speed': m_speed,
        # 辅助: 只在运动帧上平均, 与主判据等价(总路径/移动帧数 = 平均移动帧速度)
        'speed_loco': float((step[moving] * FPS).mean()) if moving.any() else 0.0,
        'cadence': count_steps(cache['contact'], s, e) / dt,
        'energy': motion_energy(joints, s, e),
        # 辅助: 移动帧占全部帧的比例
        'moving_ratio': float(moving.mean()),
    }


def metric_leg_speed(joints, s, e, cache):
    """walk/run: 双脚相对根节点的运动速度 (m/s), 恒正

    去掉根节点全局平移, 只看双脚(左右脚, 关节10/11)相对身体的速度。
    跑和走的核心差异在末端(脚)的摆动频率和幅度, 髋/膝会稀释信号。
    原地跑步(根节点几乎不动)也能被正确测量为"快", 不会被误判为"慢"。
    排除双脚静止帧(双脚平均速度 < MOVE_EPS m/s), 避免站立帧稀释均值。
    """
    leg = joints[s:e, LEG_JOINTS] - joints[s:e, 0:1]               # (T,2,3) 相对根节点
    step = np.linalg.norm(leg[1:] - leg[:-1], axis=-1)             # (T-1,2)
    frame_vel = step.mean(axis=-1) * FPS                            # (T-1,) 双脚平均逐帧速度
    moving = frame_vel > MOVE_EPS
    m_leg_speed = float(frame_vel[moving].mean()) if moving.any() else 0.0
    return {'leg_speed': m_leg_speed}


def metric_facing_disp(joints, s, e, cache):
    """forward/backward: 走的方向与"自身当前朝向"的夹角 —— 纯方向量, 不掺速度

    主判据 m_align = Sum<dp_t, f_t> / Sum|dp_t|, 即按位移量加权的方向余弦, [-1,1]:
        +1 完全朝着自己面朝的方向走(forward)
         0 纯侧向平移
        -1 完全背着自己面朝的方向走(backward, 真倒着走)
    分母是走过的总里程, 把"走多快"约掉 —— 走得远不等于方向更准, 那是 speed 类的事。

    用逐帧的 forward[t] 而非第 0 帧的 F0, 所以"转身 180 度再往前走"会被正确
    计为 forward 而不是 backward(转完身后速度与朝向仍同向)。
    但它管不了"朝向是否留在原始前方", 那需要配合 heading_dev 一起判。
    """
    dt = cache['dt']
    disp = cache['disp']
    fwd = cache['forward'][s:e - 1]
    proj = float(np.sum(disp * fwd))
    path = cache['path']
    m_align = proj / path if path > 1e-6 else 0.0

    # 逐帧方向: 每帧位移与当前朝向的点积, >0=向前走, <0=向后走
    frame_dot = np.sum(disp * fwd, axis=-1)      # (N,) 逐帧的 <dp_t, forward_t>
    n_frames = len(frame_dot)
    # 只统计根节点真正在移动的帧: 逐帧位移模长 > 阈值 (排除静止帧/纯转身帧)
    frame_speed = np.linalg.norm(disp, axis=-1) * FPS   # (N,) 逐帧瞬时速度 m/s
    moving = frame_speed > MOVE_EPS                       # 速度 > 0.1 m/s 才算"在走"
    n_moving = max(moving.sum(), 1)
    # 方向比例: 只在移动帧上统计
    fwd_pct = float((frame_dot[moving] > 0).mean() * 100) if moving.any() else 0.0
    bwd_pct = float((frame_dot[moving] < 0).mean() * 100) if moving.any() else 0.0
    # 严格逐帧正确率: 只要求所有移动帧的方向都对, 静止帧/转身帧不管
    all_fwd = int(np.all(frame_dot[moving] > 0)) if moving.any() else 0
    all_bwd = int(np.all(frame_dot[moving] < 0)) if moving.any() else 0

    # 朝向相对窗口起始帧偏离了多少度, 值域 [0,180]。
    # 注意必须用"首末朝向夹角"而不是累计转角 abs_yaw: 后者转满 360 度会得到 360,
    # 但实际朝向已回到原方向, 用它判"faces forward 是否保持"会误杀。
    f0, f1 = cache['forward'][s], cache['forward'][e - 1]
    heading_dev = float(np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1.0, 1.0))))

    # 逐帧朝向偏离均值: 每一帧偏离初始朝向的角度取平均, 值域 [0,180]。
    # 相比 heading_dev(只看首末):
    #   - 弧形倒走(缓慢均匀偏转) mean_heading_dev ≈ 总偏转/2, 不会误杀
    #   - 转身作弊(前几帧突然转 180°) mean_heading_dev 仍很大, 能被抓住
    #   - 直线倒走 mean_heading_dev ≈ 0, 与 heading_dev 一致
    fwd_all = cache['forward'][s:e]
    f0 = cache['forward'][s]
    dev_per_frame = np.degrees(np.arccos(np.clip(np.sum(fwd_all * f0, axis=-1), -1.0, 1.0)))
    mean_heading_dev = float(dev_per_frame.mean())

    return {
        # 主判据: 方向余弦 [-1,1]
        'm_align': m_align,
        # 展示用: 走向与朝向的夹角(度), 0 = 完全朝前走, 180 = 完全倒着走
        'align_angle': float(np.degrees(np.arccos(np.clip(m_align, -1.0, 1.0)))),
        # 辅助: 朝向偏离度(首末夹角), 文本说 "facing forward" 时应接近 0 -> 用于排除转身情形
        'heading_dev': heading_dev,
        # 辅助: 逐帧朝向偏离均值, 比 heading_dev 更能区分"弧形倒走"和"转身作弊"
        'mean_heading_dev': mean_heading_dev,
        # 辅助: 未归一化版本(m/s), 说明幅度
        'm_fwd': proj / dt,
        # 辅助: 用第 0 帧固定前轴(等价于 z 坐标差), 会被"转身后往前走"骗过, 仅作对照
        'fwd_disp_ref': float(np.dot(cache['p'][e - 1] - cache['p'][s], cache['forward'][s])) / dt,
        # 辅助: 累计转角绝对值(仅供参考, 判朝向保持请用 heading_dev)
        'abs_yaw': float(abs(np.degrees(cache['yaw_delta'][s:e - 1].sum()))),
        # 辅助: 逐帧方向的比例 —— 所有帧中有多少比例的帧在向前/向后走
        'fwd_frame_pct': fwd_pct,
        'bwd_frame_pct': bwd_pct,
        # 辅助: 所有帧方向都一致的motion (全部向前=1, 全部向后=1)
        'all_fwd': all_fwd,
        'all_bwd': all_bwd,
    }


def metric_turn_rate(joints, s, e, cache):
    """clockwise/ccw: 每秒朝向转角(deg/s)。+ = 逆时针(俯视)

    逐帧增量累加, 不用首末夹角 —— 后者落在 (-180,180], 转满一圈会被算成 0。
    """
    dt = cache['dt']
    psi = cache['yaw_delta'][s:e - 1].sum()

    # 辅助: 轨迹绕转角, 把朝向换成速度方向, 只在有效运动帧累加
    disp = cache['disp']
    speed = np.linalg.norm(disp, axis=-1) * FPS
    valid = speed > MOVE_EPS
    if valid.sum() >= 2:
        u = normalize_vec(disp)
        d_path = signed_yaw_increments(u)
        m = valid[:-1] & valid[1:]
        psi_path = float(d_path[m].sum())
    else:
        psi_path = 0.0

    return {
        'm_yaw': float(np.degrees(psi)) / dt,
        'yaw_path': float(np.degrees(psi_path)) / dt,
    }


def metric_hand_height(joints, s, e, cache):
    """left/right hand: 左右手腕的高度差。+ = 左手更高

    主判据 m_hand = mean_t( y[L_WRIST] - y[R_WRIST] ), 单位 m。
    取差值而非单侧绝对高度: 差值自动消掉"整体举手幅度"这个共同因素, 只留下"举的是哪只手",
    而且对人物站姿高低不敏感(HumanML3D 用统一骨架, 但差值更稳)。
    """
    w = joints[s:e]
    yl = w[:, L_WRIST, 1]
    yr = w[:, R_WRIST, 1]
    y_root = w[:, 0, 1]
    diff = yl - yr
    return {
        # 主判据: 左右手腕平均高度差(m), + = 左手更高
        'm_hand': float(diff.mean()),
        # 辅助: 峰值高度差, 举手是瞬时动作时均值会被稀释
        'hand_diff_max': float(diff.max()),
        # 辅助: 各手腕相对根节点的平均高度, 看是否真的"举高了"而不是两只手都垂着
        'hand_l_rel': float((yl - y_root).mean()),
        'hand_r_rel': float((yr - y_root).mean()),
        # 辅助: 举得最高时相对根节点的高度
        'hand_l_rel_max': float((yl - y_root).max()),
        'hand_r_rel_max': float((yr - y_root).max()),
    }


def motion_energy(joints, s, e):
    """去掉全局水平平移后的全身逐帧运动量(m/s), 可捕捉"原地快速踏步" """
    j = joints[s:e].astype(np.float64).copy()
    j = j - hproj(j[:, 0:1])
    return float(np.linalg.norm(j[1:] - j[:-1], axis=-1).mean()) * FPS


def compute_all_metrics(joints, s, e):
    """统一入口: 一次算完四类主判据 + 全部辅助量。joints: (T,22,3)"""
    right, forward, left = body_frames(joints)
    p = root_horizontal(joints)
    disp = frame_disp(p, s, e)
    cache = {
        'right': right, 'forward': forward, 'left': left,
        'p': p, 'disp': disp,
        'dt': (e - s) / FPS,
        'path': float(np.linalg.norm(disp, axis=-1).sum()),
        'yaw_delta': signed_yaw_increments(forward),
        'contact': foot_contact_from_joints(joints),
    }
    out = {'path_len': cache['path'], 'no_move': int(cache['path'] < NO_MOVE_PATH)}
    out.update(metric_lateral(joints, s, e, cache))
    out.update(metric_speed(joints, s, e, cache))
    out.update(metric_leg_speed(joints, s, e, cache))
    out.update(metric_facing_disp(joints, s, e, cache))
    out.update(metric_turn_rate(joints, s, e, cache))
    out.update(metric_hand_height(joints, s, e, cache))
    return out


# ==================== 汇总统计 (md §2) ====================
def summarize_pair(vals_a, vals_b, no_move_a=None, no_move_b=None, signed=True,
                   filter_no_move=True, angle_a=None, angle_b=None,
                   con_a=None, con_b=None, con_max=None):
    """对内两条文本的汇总: 均值 / diff / 倍率 / paired / 绝对符号准确率 / 静止率

    vals_a, vals_b: (n,) 同一组配对噪声下 A / B 两条文本的主判据值

    filter_no_move: 归一化的方向判据(余弦)分母是位移模长/里程, 静止样本会得到纯噪声,
                    必须整体剔除(均值和准确率都剔)。而 speed 类"静止"本身就是慢的信息、
                    cw_ccw 类"原地转身"是合法实现, 这两类不能剔。

    两个准确率互补, 缺一不可:
      paired      配对差值符号正确的比例 —— 衡量"相对区分度"。但它对绝对方向不敏感:
               若某模型对 ccw 给 -0.4、对 cw 给 -10.8(两侧都在顺时针转), 差值方向仍"对",
               paired 会虚高到 92%, 掩盖了"模型根本不懂 counter 前缀"这一事实。
      sign_acc A 侧 m>0 且 B 侧 m<0 的比例 —— 衡量"绝对方向是否正确", 正是 paired 漏掉的部分。
               仅对有天然 0 边界的类别有意义(left/right, fwd/bwd, cw/ccw), speed 类恒正故跳过。
    """
    a_all = np.asarray(vals_a, dtype=np.float64)
    b_all = np.asarray(vals_b, dtype=np.float64)

    if no_move_a is not None:
        valid = ~(np.asarray(no_move_a, bool) | np.asarray(no_move_b, bool))
    else:
        valid = np.ones(len(a_all), dtype=bool)
    keep = valid if filter_no_move else np.ones(len(a_all), dtype=bool)
    if not keep.any():
        keep = np.ones(len(a_all), dtype=bool)

    a, b = a_all[keep], b_all[keep]
    val_diff = a - b
    diff_rate = float((val_diff > 0).mean())

    if signed:
        sign_acc_a = float((a > 0).mean())
        sign_acc_b = float((b < 0).mean())
        sign_acc = 0.5 * (sign_acc_a + sign_acc_b)
    else:
        sign_acc_a = sign_acc_b = sign_acc = float('nan')

    if angle_a is not None:
        ang_a = float(np.asarray(angle_a, dtype=np.float64)[keep].mean())
        ang_b = float(np.asarray(angle_b, dtype=np.float64)[keep].mean())
    else:
        ang_a = ang_b = float('nan')

    ratio = float(a.mean() / b.mean()) if abs(b.mean()) > 1e-6 else float('nan')

    # strict-accuracy: 符号正确 **且** 满足附加约束(如 fwd_bwd 要求朝向留在原始前方)。
    # dir_acc 只看方向符号, 抓不到"转身后倒着走"这种情形, 必须用约束量补。
    if con_a is not None and con_max is not None and signed:
        ca = np.asarray(con_a, dtype=np.float64)[keep]
        cb = np.asarray(con_b, dtype=np.float64)[keep]
        strict_a = float(((a > 0) & (ca < con_max)).mean())
        strict_b = float(((b < 0) & (cb < con_max)).mean())
        strict_acc = 0.5 * (strict_a + strict_b)
        viol_a = float((ca >= con_max).mean())
        viol_b = float((cb >= con_max).mean())
    else:
        strict_a = strict_b = strict_acc = viol_a = viol_b = float('nan')

    return {
        'n': int(len(a)),
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'std_a': float(a.std()), 'std_b': float(b.std()),
        'angle_a': ang_a, 'angle_b': ang_b,
        'diff': float(a.mean() - b.mean()),           # 两侧均值差 (原 gap)
        'ratio': ratio,
        'diff_pct': diff_rate,                         # 配对差值正确率 (原 diff)
        'sign_acc': sign_acc, 'sign_acc_a': sign_acc_a, 'sign_acc_b': sign_acc_b,
        'strict_acc': strict_acc, 'strict_acc_a': strict_a, 'strict_acc_b': strict_b,
        'violate_a': viol_a, 'violate_b': viol_b,
        'n_valid': int(keep.sum()),
        'no_move_rate': float(1.0 - valid.mean()),
        'sign_flip': int(a.mean() > 0 > b.mean()),   # A 正 B 负 = 干净的符号翻转
    }


# ==================== 模型与生成 ====================
def load_norm():
    return np.load(MEAN_PATH), np.load(STD_PATH)


def build_model(args, ckpt_path, use_lora):
    """按 sample-lora.py:54 的逻辑构建模型; add_clip_lora 影响网络结构, 必须按模型切换"""
    from models.mdm_bert.mdm_bert import MDMBERT
    from utils.model_util import get_mdm_bert_args, create_gaussian_diffusion_simple
    from utils.lora_util import load_lora_mdm_for_eval
    from utils.mask_utils import load_ckpt

    args.add_clip_lora = use_lora
    net = MDMBERT(**get_mdm_bert_args(args, 'mdm_bert'))
    if use_lora:
        load_lora_mdm_for_eval(net, ckpt_path)
    else:
        load_ckpt(net, ckpt_path, key=None, strict=False)

    diffusion = create_gaussian_diffusion_simple(args, net, 'mdm_bert')
    net.cuda()
    net.eval()
    return net, diffusion


def generate(diffusion, text, n, seed, length=GEN_LENGTH):
    """生成 n 个 motion。返回归一化的 (n, length, 263) numpy

    配对噪声: p_sample_loop 不接受 noise 参数(gaussian_diffusion_simple.py:829 内部
    torch.randn), 且 p_sample 每步还有 randn_like。故在调用前重设种子, 使初始噪声和
    整条采样轨迹的每步噪声都完全一致 —— 比传 noise= 配得更彻底。
    """
    from utils.mask_utils import generate_src_mask

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    real_length = torch.Tensor([length]).int().cuda()
    real_mask = generate_src_mask(length, real_length).repeat(n, 1)
    model_kwargs = {'clip_text': (text,) * n, 'real_mask': real_mask}

    sample = diffusion.p_sample_loop(None, model_kwargs=model_kwargs, batch_size=n)
    return sample.detach().cpu().numpy()


def denorm_to_joints(sample, mean, std):
    """反归一化 + recover_from_ric。(n,T,263) -> (n,T,22,3)"""
    motion = sample * std + mean
    return recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()


def skating(joints):
    """foot skating ratio, 复用 utils/metrics.py:480。joints: (n,T,22,3) -> (n,)"""
    from utils.metrics import calculate_skating_ratio
    m = torch.from_numpy(joints).float().permute(0, 2, 3, 1)   # (n,22,3,T)
    ratio, _ = calculate_skating_ratio(m)
    return np.asarray(ratio)


def cache_path(model_name, cat_name, tid, side, text):
    """缓存文件名带文本的短哈希 —— 改了 prompt 会自动生成新文件, 旧文件原样保留不覆盖"""
    tag = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f'{model_name}_{cat_name}_{tid}_{side}_{tag}.npy')


def all_cached(model_name):
    """该模型的全部样本是否已缓存 —— 若是则无需加载模型, 重算指标不占 GPU"""
    return all(os.path.exists(cache_path(model_name, cat['name'], tid, side, text))
               for cat in CATEGORIES
               for tid, pair in enumerate(cat['pairs'])
               for side, text in zip(('a', 'b'), pair))


def get_joints(diffusion, model_name, cat_name, tid, side, text, seed, mean, std):
    """带磁盘缓存的生成, 便于反复调指标而不重跑扩散"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(model_name, cat_name, tid, side, text)
    if os.path.exists(path):
        return np.load(path)
    sample = generate(diffusion, text, N_SAMPLES, seed)
    joints = denorm_to_joints(sample, mean, std)
    np.save(path, joints)
    return joints


# ==================== GT 符号校准 (md §7) ====================
# 已严格人工核对过的 GT 样本(文件内所有 caption 都含该关键词、不含反义词、且非肢体用法)。
# 注: 010020 "stumbles in a clockwise circular motion" 虽符号正确但 m_yaw 仅 -0.37(人是踉跄
# 绕圈, 身体朝向几乎没转, yaw_path=-36.6), 太贴近 0 不适合做断言样本, 故未纳入。
GT_CALIB = {
    'left':     ['000365', '000879', '002141', '002427'],
    'right':    ['000390', '000407', '000463', '000834', '001081'],
    'backward': ['000028', '000109', '000144', '000178', '000267', '000282'],
    # 左右转(借 left/right 的样本交叉验证 yaw 符号)
    'turn_left':  ['000879'],   # "a person walks and turns left"
    'turn_right': ['000834'],   # "a person walks forwards and turns right"
    # 真正含 clockwise / counterclockwise 的样本
    'ccw': ['000212', '003456', '009648', '010002', '011492'],
    'cw':  ['001236', '002448', '003329', '006926', '007662', '008872', '010378'],
    # 抬左/右手的样本(只取幅度明确的, 排除"把手放台面上""双手同时举起"这类边界样本)
    'hand_l': ['001551', '001923', '002363', '005627', '005754'],
    'hand_r': ['000049', '000177', '001391', '001424', '001540'],
}


def load_gt_joints(name):
    return np.load(os.path.join(GT_JOINT_DIR, name + '.npy')).reshape(-1, 22, 3)


def calibrate_on_gt():
    """用 GT motion 跑同一套判据并断言符号, 防止左右/顺逆整体反向"""
    print('=' * 78)
    print('GT 符号校准 (md §7)')
    print('=' * 78)

    # --- 参考系本身 ---
    j = load_gt_joints(GT_CALIB['left'][0])
    right, forward, _ = body_frames(j)
    print(f"\n[参考系] right[0]={np.round(right[0], 3)}  forward[0]={np.round(forward[0], 3)}")
    print("         预期 right=[-1,0,0] (即 +X 指向人体左侧), forward=[0,0,+1]")
    assert right[0][0] < -0.9, 'right 轴方向与预期不符, 后续所有左右符号都会反!'

    def run(names, key):
        out = []
        for nm in names:
            jj = load_gt_joints(nm)
            T = len(jj)
            m = compute_all_metrics(jj, 0, T)
            out.append((nm, m[key], T))
        return out

    checks = [
        ('m_lat  > 0  (走向左)',  GT_CALIB['left'],     'm_lat',  +1),
        ('m_lat  < 0  (走向右)',  GT_CALIB['right'],    'm_lat',  -1),
        ('m_fwd  < 0  (倒着走)',  GT_CALIB['backward'], 'm_fwd',  -1),
        ('m_yaw  > 0  (左转)',    GT_CALIB['turn_left'],  'm_yaw', +1),
        ('m_yaw  < 0  (右转)',    GT_CALIB['turn_right'], 'm_yaw', -1),
        ('m_yaw  > 0  (逆时针)',  GT_CALIB['ccw'],      'm_yaw',  +1),
        ('m_yaw  < 0  (顺时针)',  GT_CALIB['cw'],       'm_yaw',  -1),
        ('m_hand > 0  (抬左手)',  GT_CALIB['hand_l'],   'm_hand', +1),
        ('m_hand < 0  (抬右手)',  GT_CALIB['hand_r'],   'm_hand', -1),
    ]
    all_ok = True
    for title, names, key, want in checks:
        res = run(names, key)
        ok = sum(1 for _, v, _ in res if np.sign(v) == want)
        flag = 'OK ' if ok == len(res) else '!! '
        all_ok &= (ok == len(res))
        print(f"\n[{flag}] {title:26s}  {ok}/{len(res)} 通过")
        for nm, v, T in res:
            print(f"        {nm}  {key}={v:+7.3f}   (T={T})")

    print('\n' + '=' * 78)
    print('全部符号校准通过' if all_ok else '存在未通过项, 结论可能整体反向, 必须先修正!')
    print('=' * 78)
    return all_ok


# ==================== 主流程 ====================
def format_num(v, nd=2):
    return 'nan' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:+.{nd}f}'


WINDOW_DESC = {'3s': '前3秒 = 0~60帧', '6s': '前6秒 = 0~120帧', 'full': '全程9.8秒 = 0~196帧'}


def metric_legend():
    """指标含义总说明, 全文只打印一次"""
    return """
每一类的打印内容怎么看 (看不懂时回来查这里)
------------------------------------------------------------------------------
  每行的 "+x.xx / -x.xx" 是什么:
      斜杠左边 是喂 "left" / "quickly" / "counterclockwise" / "left hand" 等等文本时,
               100个motion的主判据均值。
      斜杠右边 是喂 "right" / "slowly" / "clockwise" / "right hand" 等等反义文本时,
               100个motion的主判据均值。
      正负号的含义见每类自己的「符号约定」行。

  [对]/[错]     该侧均值的符号是否正确。A侧期望为正, B侧期望为负(一正一负 = 方向跟着关键词翻了)。
                 speed 类数值恒正, 只比大小, 不标注对错。

  diff          两侧均值的差 = A侧均值 - B侧均值。diff 越大 = 换个词动作变得越厉害。

  diff%         100组配对噪声中, "A侧数值 > B侧数值" 的比例。瞎猜 = 50%。
                衡量换关键词后数值的相对变化。局限: 两侧同号时这个数字仍然可能很高。
                例: 对 counterclockwise 给 -0.4、对 clockwise 给 -10.8(两次都在顺时针转),
                但 -0.4 > -10.8 成立, diff% 仍高达 92%。所以它必须配合 correct 一起看。

  correct       (A侧符号正确的比例 + B侧符号正确的比例) / 2。瞎猜 = 50%。
                例: 要求举左手, 左手真的比右手高 -> 这一侧"正确"。
                括号里 (A xx% / B xx%) 是两侧各做对了多少。
                **这是最该看的指标** —— 只问方向有没有跟着关键词走，不问走了多远。

  sign-flip     出现这个标记 -> A侧均值为正 且 B侧均值为负, 即"说左就往左、说右就往右"。
                但这是平均值, 不代表每个样本都对了 —— 仍需看 correct 来确认。

  怎么一眼看好坏:
      (1) 先看 correct 够不够高  (2) 看有没有 sign-flip  (3) 最后比 diff 谁大。
------------------------------------------------------------------------------"""


def verdict_text(r, signed):
    """给该模型在该类上的一句话判定

    注意不能只看均值的符号翻转: 均值翻转只说明"平均而言两侧朝相反方向", 但逐样本可能
    有很大比例是错的(例: 均值 +0.29/-0.30 看似完美翻转, correct 却只有 65%)。
    所以判定必须结合 correct。
    """
    if not signed:
        return f">> A侧是B侧的 {r['ratio']:.2f} 倍" if not np.isnan(r['ratio']) else ''

    da = r['sign_acc']
    if not r['sign_flip']:
        same = '都为正' if r['mean_a'] > 0 and r['mean_b'] > 0 else '都为负'
        return f'>> 两侧同号({same}) -> 不管文本说哪个词都朝同一方向做, 未真正区分关键词'
    sa = r.get('strict_acc', float('nan'))
    if not np.isnan(sa) and da - sa > 0.15:
        return (f'>> correct {100*da:.0f}% 看似很好, 但加上附加约束后只剩 {100*sa:.0f}% '
                f'-> 大量样本靠"违反约束"达成方向, 不能算真正遵循文本')
    if da >= 0.95:
        return f'>> 均值 sign-flip, 且逐样本 correct {100*da:.0f}% -> 稳定地区分了这对关键词'
    if da >= 0.80:
        return f'>> 均值 sign-flip, 逐样本 correct {100*da:.0f}% -> 基本区分了这对关键词, 有少量样本方向错'
    if da >= 0.70:
        return f'>> 均值 sign-flip, 但逐样本 correct 仅 {100*da:.0f}% -> 区分不稳定, 相当一部分样本方向是错的'
    return (f'>> 均值虽 sign-flip, 但逐样本 correct 仅 {100*da:.0f}% (瞎猜是50%) '
            f'-> 实际几乎没区分, 只是平均下来略偏')


def emit_rebuttal_text(summary, cat, window_name):
    """打印某一类的完整解读块: 判据定义 + 符号约定 + 期望 + 两模型逐侧数值 + 判定"""
    s = {m: summary[(m, cat['name'], window_name)] for m, _, _ in MODELS
         if (m, cat['name'], window_name) in summary}
    if not s:
        return '  (no data)'
    a, b, unit, signed = cat['label_a'], cat['label_b'], cat['unit'], cat['signed']

    L = [f"  主判据: {cat['metric']}  (单位 {unit})",
         f"  含义:   {cat['desc']}"]
    if signed:
        L.append(f'  符号约定: 数值 > 0 表示朝 "{a}" 的方向;  数值 < 0 表示朝 "{b}" 的方向')
        L.append(f'  期望结果: 喂 "{a}" 的文本 -> 正数;  喂 "{b}" 的文本 -> 负数  (即一正一负)')
    else:
        L.append('  符号约定: 数值恒为正, 只比大小')
        L.append(f'  期望结果: 喂 "{a}" 的文本的数值  >  喂 "{b}" 的文本的数值')
    L.append(f"  统计窗口: {window_name} ({WINDOW_DESC.get(window_name, '')})")
    L.append(f'  样本量:   5个文本模板 x 20个样本 = 100 配对/模型 (两模型共享同一组噪声)')
    if cat.get('angle_desc'):
        L.append(f"  角度含义: {cat['angle_desc']}")
    if cat.get('constraint'):
        L.append(f"  附加约束: {cat['constraint']['desc']}")
        L.append(f"            -> strict-accuracy = 方向符号正确 **且** 满足该约束的比例")
    L.append(f"  数值含义: 下面的每个 +x.xx / -x.xx 是喂对应文本后 100个motion的主判据均值(单位 {unit})。")
    L.append(f"            斜杠左边 = 喂 \"{a}\" 文本 -> 这 100个的均值; 斜杠右边 = 喂 \"{b}\" 文本 -> 那 100个的均值。")
    L.append(f"            符号(正/负)决定方向, 见上方「符号约定」。gap = 两侧均值之差。")
    L.append(f'  文本示例: A侧 "{cat["pairs"][0][0]}"')
    L.append(f'            B侧 "{cat["pairs"][0][1]}"')
    L.append('')

    for mname, _, _ in MODELS:
        r = s.get(mname)
        if r is None:
            continue
        ok_a = '' if not signed else ('[对]' if r['mean_a'] > 0 else '[错, 应为正]')
        ok_b = '' if not signed else ('[对]' if r['mean_b'] < 0 else '[错, 应为负]')
        fmt = '{:+.0f}' if (cat['angle_key'] == 'disp_angle') else '{:.0f}'
        ang_a = '' if np.isnan(r['angle_a']) else '  (角度 ' + fmt.format(r['angle_a']) + '度)'
        ang_b = '' if np.isnan(r['angle_b']) else '  (角度 ' + fmt.format(r['angle_b']) + '度)'
        L.append(f'  {mname:5s}  A侧 "{a}" 喂进去 -> {r["mean_a"]:+7.3f} {unit} {ok_a}{ang_a}')
        L.append(f'  {"":5s}  B侧 "{b}" 喂进去 -> {r["mean_b"]:+7.3f} {unit} {ok_b}{ang_b}')
        acc = ('' if np.isnan(r['sign_acc']) else
               f' | correct {100*r["sign_acc"]:.0f}% (A侧 {100*r["sign_acc_a"]:.0f}% / B侧 {100*r["sign_acc_b"]:.0f}%)')
        L.append(f'  {"":5s}  diff {r["diff"]:.3f} {unit} | diff% {100*r["diff_pct"]:.0f}%{acc}'
                 f' | 有效样本 {r["n_valid"]}/{N_SAMPLES*len(cat["pairs"])}')
        if not np.isnan(r['strict_acc']):
            L.append(f'  {"":5s}  constraint-violated: A侧 {100*r["violate_a"]:.0f}% / B侧 {100*r["violate_b"]:.0f}%'
                     f'  ->  strict-accuracy {100*r["strict_acc"]:.0f}%'
                     f' (A侧 {100*r["strict_acc_a"]:.0f}% / B侧 {100*r["strict_acc_b"]:.0f}%)')
        L.append(f'  {"":5s}  {verdict_text(r, signed)}')
        L.append('')
    return '\n'.join(L)


def main():
    args.dataset_name = 't2m'
    args.modeltype = 'mdm_bert'
    args.text_encoder_type = 'clip'
    args.diffusion_steps = DIFFUSION_STEPS
    args.batch_size = N_SAMPLES
    args.no_random = True

    calibrate_on_gt()

    mean, std = load_norm()
    sample_rows = []
    summary = {}

    for model_name, ckpt, use_lora in MODELS:
        print(f'\n########## {model_name}  ({ckpt}) ##########')
        if all_cached(model_name):
            print('  全部样本已缓存, 跳过模型加载')
            diffusion = None
        else:
            _, diffusion = build_model(args, ckpt, use_lora)

        for cat_idx, cat in enumerate(CATEGORIES):
            # 每个窗口下, 把 5 个模板的 20 个样本汇集成 n=100
            pooled = {w[0]: {'a': [], 'b': [], 'nm_a': [], 'nm_b': [], 'ang_a': [], 'ang_b': [],
                             'con_a': [], 'con_b': [], 'fwd_a': [], 'fwd_b': [],
                             'bwd_a': [], 'bwd_b': [], 'all_fwd_a': [], 'all_fwd_b': [],
                             'all_bwd_a': [], 'all_bwd_b': []} for w in EVAL_WINDOWS}
            skate_a, skate_b = [], []

            for tid, (text_a, text_b) in enumerate(cat['pairs']):
                # 对内两条文本 + 两个模型共用同一 seed -> 完全配对。
                # 不能用 hash(str): Python3 的字符串 hash 受 PYTHONHASHSEED 随机化, 跨进程不稳定。
                seed = BASE_SEED + cat_idx * 100 + tid
                joints = {}
                for side, text in (('a', text_a), ('b', text_b)):
                    joints[side] = get_joints(diffusion, model_name, cat['name'], tid, side,
                                              text, seed, mean, std)
                    (skate_a if side == 'a' else skate_b).append(skating(joints[side]))

                for wname, s, e in EVAL_WINDOWS:
                    for side, text in (('a', text_a), ('b', text_b)):
                        for i in range(N_SAMPLES):
                            m = compute_all_metrics(joints[side][i], s, e)
                            pooled[wname][side].append(m[cat['metric']])
                            pooled[wname][f'nm_{side}'].append(m['no_move'])
                            if cat['angle_key']:
                                pooled[wname][f'ang_{side}'].append(m[cat['angle_key']])
                            if cat['constraint']:
                                pooled[wname][f'con_{side}'].append(m[cat['constraint']['key']])
                            # 逐帧方向比例 (仅 fwd_bwd 类有意义, 但所有类都收集, 不碍事)
                            pooled[wname][f'fwd_{side}'].append(m.get('fwd_frame_pct', 0.0))
                            pooled[wname][f'bwd_{side}'].append(m.get('bwd_frame_pct', 0.0))
                            pooled[wname][f'all_fwd_{side}'].append(m.get('all_fwd', 0))
                            pooled[wname][f'all_bwd_{side}'].append(m.get('all_bwd', 0))
                            sample_rows.append(dict(
                                model=model_name, category=cat['name'], template_id=tid,
                                side=side, keyword=cat['label_a'] if side == 'a' else cat['label_b'],
                                text=text, sample_idx=i, window=wname,
                                main_metric=cat['metric'], main_value=m[cat['metric']], **m))

                print(f'  [{cat["name"]}] template {tid} done')

            skate_a_mean = float(np.concatenate(skate_a).mean())
            skate_b_mean = float(np.concatenate(skate_b).mean())
            for wname, _, _ in EVAL_WINDOWS:
                r = summarize_pair(pooled[wname]['a'], pooled[wname]['b'],
                                   pooled[wname]['nm_a'], pooled[wname]['nm_b'],
                                   signed=cat['signed'],
                                   filter_no_move=cat['filter_no_move'],
                                   angle_a=pooled[wname]['ang_a'] or None,
                                   angle_b=pooled[wname]['ang_b'] or None,
                                   con_a=pooled[wname]['con_a'] or None,
                                   con_b=pooled[wname]['con_b'] or None,
                                   con_max=cat['constraint']['max'] if cat['constraint'] else None)
                r['skating'] = 0.5 * (skate_a_mean + skate_b_mean)
                r['skating_a'] = skate_a_mean
                r['skating_b'] = skate_b_mean
                # 逐帧方向比例: A侧所有样本的平均 forward%, B侧所有样本的平均 backward%
                r['fwd_frame_pct_a'] = float(np.mean(pooled[wname]['fwd_a']))
                r['fwd_frame_pct_b'] = float(np.mean(pooled[wname]['fwd_b']))
                r['bwd_frame_pct_a'] = float(np.mean(pooled[wname]['bwd_a']))
                r['bwd_frame_pct_b'] = float(np.mean(pooled[wname]['bwd_b']))
                # 严格逐帧正确率: A侧"所有帧都在向前走"的比例, B侧"所有帧都在向后走"的比例
                r['frame_strict_a'] = float(np.mean(pooled[wname]['all_fwd_a']))
                r['frame_strict_b'] = float(np.mean(pooled[wname]['all_bwd_b']))
                r['frame_strict_acc'] = 0.5 * (r['frame_strict_a'] + r['frame_strict_b'])
                summary[(model_name, cat['name'], wname)] = r

    # ---- 写 CSV ----
    if sample_rows:
        keys = list(sample_rows[0].keys())
        with open(OUT_SAMPLES_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sample_rows)

    with open(OUT_SUMMARY_CSV, 'w', newline='') as f:
        cols = ['model', 'category', 'window', 'metric', 'unit', 'n', 'mean_a', 'std_a',
                'mean_b', 'std_b', 'angle_a', 'angle_b', 'diff', 'ratio', 'diff_pct', 'sign_acc',
                'sign_acc_a', 'sign_acc_b', 'strict_acc', 'strict_acc_a', 'strict_acc_b',
                'violate_a', 'violate_b', 'n_valid', 'no_move_rate', 'sign_flip', 'skating', 'skating_a', 'skating_b',
                'fwd_frame_pct_a', 'fwd_frame_pct_b', 'bwd_frame_pct_a', 'bwd_frame_pct_b',
                'frame_strict_a', 'frame_strict_b', 'frame_strict_acc']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (mname, cname, wname), r in summary.items():
            cat = next(c for c in CATEGORIES if c['name'] == cname)
            w.writerow(dict(model=mname, category=cname, window=wname,
                            metric=cat['metric'], unit=cat['unit'], **r))

    # ---- 控制台: rebuttal 可直接粘贴的文本 ----
    print('\n\n' + '=' * 78)
    print('REBUTTAL 数字')
    print('=' * 78)
    print(metric_legend())
    for k, cat in enumerate(CATEGORIES):
        print('\n' + '#' * 78)
        print(f"# 第{k+1}类  A侧 \"{cat['label_a']}\"   vs   B侧 \"{cat['label_b']}\"")
        print('#' * 78)
        print(emit_rebuttal_text(summary, cat, cat['report_window']))

    print('\n\n' + '=' * 78)
    print('全窗口对照 (同一批动作换三个统计时长各算一遍, 用于检查结论是否依赖窗口选择)')
    print('=' * 78)
    for cat in CATEGORIES:
        sign_hint = (f'  / 左边 = 喂 "{cat["label_a"]}" 文本,   / 右边 = 喂 "{cat["label_b"]}" 文本'
                     f'\n  期望: 左边数值 >0, 右边数值 <0 (一正一负 = 方向跟着关键词翻了)') if cat['signed'] else \
                    (f'  / 左边 = 喂 "{cat["label_a"]}" 文本,   / 右边 = 喂 "{cat["label_b"]}" 文本'
                     f'\n  期望: 左边数值 > 右边数值 (speed 只有大小没有方向, 只看谁更大)')
        print(f"\n{cat['name']}   主判据 {cat['metric']} (单位 {cat['unit']})")
        print(f"  {sign_hint}")
        for wname, _, _ in EVAL_WINDOWS:
            for mname, _, _ in MODELS:
                r = summary.get((mname, cat['name'], wname))
                if not r:
                    continue
                da = '' if np.isnan(r['sign_acc']) else f'  correct {100*r["sign_acc"]:3.0f}%'
                flip = '  sign-flip' if r['sign_flip'] else ''
                print(f'  {wname:5s} {mname:5s}: {r["mean_a"]:+7.2f} / {r["mean_b"]:+7.2f}'
                      f'  diff {r["diff"]:6.2f}  diff% {100*r["diff_pct"]:3.0f}%{da}{flip}')

    print('\n\n' + '=' * 78)
    print('质量守护: foot skating ratio (越低越好, 复用 utils/metrics.py:480 的标准实现)')
    print('=' * 78)
    for cat in CATEGORIES:
        line = f"  {cat['name']:12s}"
        for mname, _, _ in MODELS:
            r = summary.get((mname, cat['name'], cat['report_window']))
            if r:
                line += f'   {mname} {r["skating"]:.4f} (A:{r["skating_a"]:.4f} B:{r["skating_b"]:.4f})'
        print(line)

    print(f'\n已写出 {OUT_SAMPLES_CSV} (逐样本) 和 {OUT_SUMMARY_CSV} (汇总)')

    # ==================== 逐类详细解读 (rebuttal 可直接粘贴) ====================
    print('\n\n')
    print('#' * 78)
    print('# 逐类详细解读 —— 供 rebuttal 直接引用')
    print('#' * 78)

    # ---- 第1类: left / right ----
    cat = CATEGORIES[0]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第一类：左/右行走方向 (left vs right)
{'='*78}

【测试目的】给定含 "left" 或 "right" 的文本, 观察生成的人是否真的朝对应方向行走。

【范例文本】
  说"向左": "{cat['pairs'][0][0]}"
  说"向右": "{cat['pairs'][0][1]}"
  (共5组文本模板, 每组生成20个motion, 总计每侧100个样本; 两侧共享同一组随机噪声)

【核心指标】m_lat_cos (方向余弦, 无量纲, 值域 [-1, +1])
  计算方法: 取第0帧到第120帧(6秒)之间人体根节点的净位移向量, 投影到第0帧身体的"左"轴上,
           然后除以位移模长。只衡量"方向纯不纯", 不衡量走了多远。
  含义: +1 = 完全朝正左走, -1 = 完全朝正右走, 0 = 朝正前或正后走(没有横向分量)。

【MDM 结果】
  说"向左"的100个motion, m_lat_cos 均值 = {r_mdm["mean_a"]:+.3f} (角度约 {r_mdm["angle_a"]:+.0f}°, 0°=正前 ±90°=正侧向)
  说"向右"的100个motion, m_lat_cos 均值 = {r_mdm["mean_b"]:+.3f} (角度约 {r_mdm["angle_b"]:+.0f}°)
  两侧均值一正一负 (sign-flip): {"是" if r_mdm["sign_flip"] else "否"} ← 平均而言模型分清了左右
  逐样本方向正确率 (correct): {100*r_mdm["sign_acc"]:.0f}% (瞎猜基线=50%)
    - A侧"说左往左"的比例: {100*r_mdm["sign_acc_a"]:.0f}%
    - B侧"说右往右"的比例: {100*r_mdm["sign_acc_b"]:.0f}%
  diff% (换词后数值变对的配对比例):{100*r_mdm["diff_pct"]:.0f}% (瞎猜基线=50%)

【LeGO 结果】
  说"向左"的100个motion, m_lat_cos 均值 = {r_lego["mean_a"]:+.3f} (角度约 {r_lego["angle_a"]:+.0f}°)
  说"向右"的100个motion, m_lat_cos 均值 = {r_lego["mean_b"]:+.3f} (角度约 {r_lego["angle_b"]:+.0f}°)
  两侧均值一正一负 (sign-flip): {"是" if r_lego["sign_flip"] else "否"} ← 平均而言模型分清了左右
  逐样本方向正确率 (correct): {100*r_lego["sign_acc"]:.0f}%
    - A侧"说左往左"的比例: {100*r_lego["sign_acc_a"]:.0f}%
    - B侧"说右往右"的比例: {100*r_lego["sign_acc_b"]:.0f}%
  diff%: {100*r_lego["diff_pct"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"向左"的100个motion, m_lat_cos 均值 = {r_lora["mean_a"]:+.3f} (角度约 {r_lora["angle_a"]:+.0f}°)
  说"向右"的100个motion, m_lat_cos 均值 = {r_lora["mean_b"]:+.3f} (角度约 {r_lora["angle_b"]:+.0f}°)
  两侧均值一正一负 (sign-flip): {"是" if r_lora["sign_flip"] else "否"} ← 平均而言模型分清了左右
  逐样本方向正确率 (correct): {100*r_lora["sign_acc"]:.0f}%
    - A侧"说左往左"的比例: {100*r_lora["sign_acc_a"]:.0f}%
    - B侧"说右往右"的比例: {100*r_lora["sign_acc_b"]:.0f}%
  diff%: {100*r_lora["diff_pct"]:.0f}%

【结论】LeGO (correct={100*r_lego["sign_acc"]:.0f}%) ≈ MDM+LeGO-CLIP ({100*r_lora["sign_acc"]:.0f}%) >> MDM ({100*r_mdm["sign_acc"]:.0f}%)。
两个 LoRA 模型都显著优于 MDM baseline。LeGO 的方向余弦接近 ±1 (几乎纯侧向走),
MDM+LeGO-CLIP 约 ±0.8 (方向较纯), 而 MDM 仅 ±0.3 (方向模糊)。
LoRA 微调 CLIP 后准确理解了 "left" 和 "right" 的空间语义。""")

    # ---- 第2类: slowly / quickly ----
    cat = CATEGORIES[1]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第二类：快/慢速度 (quickly vs slowly)
{'='*78}

【测试目的】给定含 "quickly" 或 "slowly" 的文本, 观察生成的人是否走得确实更快/更慢。

【范例文本】
  说"快": "{cat['pairs'][0][0]}"
  说"慢": "{cat['pairs'][0][1]}"
  (共5组文本模板 × 20样本 = 每侧100个motion, 两侧共享同一组随机噪声)

【核心指标】m_speed (平均速度, 单位 m/s)
  计算方法: 排除静止帧(根节点瞬时速度<0.1m/s的帧), 只在移动帧上统计:
           总路径 ÷ (移动帧数/20fps)。恒为正数, 只比大小。
  为什么排除静止帧: "走走停停"的motion如果因为停的时间长而被判为"慢", 是不公平的;
           我们只关心"真的在走的时候走了多快"。

【MDM 结果】
  说"快"的100个motion, 平均速度(仅移动帧) = {r_mdm["mean_a"]:.3f} m/s
  说"慢"的100个motion, 平均速度(仅移动帧) = {r_mdm["mean_b"]:.3f} m/s
  快/慢速度比: {r_mdm["ratio"]:.2f} 倍
  diff (快均值 - 慢均值):{r_mdm["diff"]:.3f} m/s
  diff% (换"quickly"→"slowly"后速度变慢的配对比例): {100*r_mdm["diff_pct"]:.0f}% (瞎猜基线=50%)

【LeGO 结果】
  说"快"的100个motion, 平均速度 = {r_lego["mean_a"]:.3f} m/s
  说"慢"的100个motion, 平均速度 = {r_lego["mean_b"]:.3f} m/s
  快/慢速度比: {r_lego["ratio"]:.2f} 倍
  diff: {r_lego["diff"]:.3f} m/s
  diff%: {100*r_lego["diff_pct"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"快"的100个motion, 平均速度 = {r_lora["mean_a"]:.3f} m/s
  说"慢"的100个motion, 平均速度 = {r_lora["mean_b"]:.3f} m/s
  快/慢速度比: {r_lora["ratio"]:.2f} 倍
  diff: {r_lora["diff"]:.3f} m/s
  diff%: {100*r_lora["diff_pct"]:.0f}%

【结论】MDM+LeGO-CLIP 的速度比 ({r_lora["ratio"]:.2f}x) > LeGO ({r_lego["ratio"]:.2f}x) >> MDM ({r_mdm["ratio"]:.2f}x)。
MDM+LeGO-CLIP 的 diff ({r_lora["diff"]:.3f} m/s) 是 MDM ({r_mdm["diff"]:.3f} m/s) 的 {r_lora["diff"]/r_mdm["diff"]:.1f} 倍。
预训练 LoRA CLIP + 继续训练 MDM 在速度副词理解上效果最佳。""")

    # ---- 第3类: forward / backward ----
    cat = CATEGORIES[2]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第三类：向前/向后行走, 面朝前 (forward vs backward, facing forward)
{'='*78}

【测试目的】给定"面朝前向前走"和"面朝前后倒着走"的文本, 观察生成的人是否面朝前方不动、
同时真的向前走或向后倒着走。这是最难的一类: 需要同时满足 (1) 走向对 (2) 朝向不转。

【范例文本】
  说"向前": "{cat['pairs'][0][0]}"
  说"向后": "{cat['pairs'][0][1]}"
  (共5组文本模板 × 20样本 = 每侧100个motion, 两侧共享同一组随机噪声;
   所有模板均显式包含 "facing forward", 避免歧义)

【核心指标】m_align (走向-朝向对齐余弦, 无量纲, 值域 [-1, +1])
  计算方法: 对每一帧, 计算"人体根节点位移方向"与"当前身体朝向"的点积, 然后按各帧位移量
           加权平均。用的是逐帧当前朝向(而非第0帧固定朝向), 所以转身180°再往前走,
           m_align 仍 ≈ +1(被正确判为"向前走"), 不会误判为"倒着走"。
  含义: +1 = 每一步都朝着自己面朝的方向走(向前走)
         0 = 纯侧向平移
        -1 = 每一步都背对自己面朝的方向走(向后倒着走)

【MDM 结果】
  说"面朝前向前走"的100个motion, m_align 均值 = {r_mdm["mean_a"]:+.3f}
  说"面朝前后倒着走"的100个motion, m_align 均值 = {r_mdm["mean_b"]:+.3f}
  两侧均值一正一负 (sign-flip): {"是" if r_mdm["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_mdm["sign_acc"]:.0f}%
    - A侧"说向前就向前"的比例: {100*r_mdm["sign_acc_a"]:.0f}%
    - B侧"说向后就向后"的比例: {100*r_mdm["sign_acc_b"]:.0f}%
  diff%: {100*r_mdm["diff_pct"]:.0f}%
  逐帧方向比例 (排除静止帧, 只统计根节点速度>0.1m/s的帧; 静止/原地转身帧不参与统计):
    - 喂"向前走"时, 移动帧中有多少比例在真的向前移动: {r_mdm["fwd_frame_pct_a"]:.0f}%
    - 喂"向后走"时, 移动帧中有多少比例在真的向后移动: {r_mdm["bwd_frame_pct_b"]:.0f}%
      (向后走时如果先往前走再往后走, 这个比例就会显著低于100%)
  逐帧严格正确率 (frame-strict-accuracy): {100*r_mdm["frame_strict_acc"]:.0f}%
    含义: 一个motion的所有移动帧方向都正确, 才算这个motion"对"。
    - 说"向前走", 所有移动帧都在向前(>0)的motion比例: {100*r_mdm["frame_strict_a"]:.0f}%
    - 说"向后走", 所有移动帧都在向后(<0)的motion比例: {100*r_mdm["frame_strict_b"]:.0f}%
    (静止帧/原地转身帧不参与判断; 角度倾斜没关系, 只要方向对就算对)

【LeGO 结果】
  说"面朝前向前走"的100个motion, m_align 均值 = {r_lego["mean_a"]:+.3f}
  说"面朝前后倒着走"的100个motion, m_align 均值 = {r_lego["mean_b"]:+.3f}
  两侧均值一正一负 (sign-flip): {"是" if r_lego["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lego["sign_acc"]:.0f}%
    - A侧"说向前就向前"的比例: {100*r_lego["sign_acc_a"]:.0f}%
    - B侧"说向后就向后"的比例: {100*r_lego["sign_acc_b"]:.0f}%
  diff%: {100*r_lego["diff_pct"]:.0f}%
  逐帧方向比例 (排除静止帧, 只统计根节点速度>0.1m/s的帧):
    - 喂"向前走"时, 移动帧中有多少比例在真的向前移动: {r_lego["fwd_frame_pct_a"]:.0f}%
    - 喂"向后走"时, 移动帧中有多少比例在真的向后移动: {r_lego["bwd_frame_pct_b"]:.0f}%
  逐帧严格正确率 (frame-strict-accuracy): {100*r_lego["frame_strict_acc"]:.0f}%
    - 说"向前走", 所有移动帧都在向前(>0)的motion比例: {100*r_lego["frame_strict_a"]:.0f}%
    - 说"向后走", 所有移动帧都在向后(<0)的motion比例: {100*r_lego["frame_strict_b"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"面朝前向前走"的100个motion, m_align 均值 = {r_lora["mean_a"]:+.3f}
  说"面朝前后倒着走"的100个motion, m_align 均值 = {r_lora["mean_b"]:+.3f}
  两侧均值一正一负 (sign-flip): {"是" if r_lora["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lora["sign_acc"]:.0f}%
    - A侧"说向前就向前"的比例: {100*r_lora["sign_acc_a"]:.0f}%
    - B侧"说向后就向后"的比例: {100*r_lora["sign_acc_b"]:.0f}%
  diff%: {100*r_lora["diff_pct"]:.0f}%
  逐帧方向比例 (排除静止帧, 只统计根节点速度>0.1m/s的帧):
    - 喂"向前走"时, 移动帧中有多少比例在真的向前移动: {r_lora["fwd_frame_pct_a"]:.0f}%
    - 喂"向后走"时, 移动帧中有多少比例在真的向后移动: {r_lora["bwd_frame_pct_b"]:.0f}%
  逐帧严格正确率 (frame-strict-accuracy): {100*r_lora["frame_strict_acc"]:.0f}%
    - 说"向前走", 所有移动帧都在向前(>0)的motion比例: {100*r_lora["frame_strict_a"]:.0f}%
    - 说"向后走", 所有移动帧都在向后(<0)的motion比例: {100*r_lora["frame_strict_b"]:.0f}%

【结论】correct 维度: LeGO ({100*r_lego["sign_acc"]:.0f}%) ≈ MDM+LeGO-CLIP ({100*r_lora["sign_acc"]:.0f}%) > MDM ({100*r_mdm["sign_acc"]:.0f}%)。
strict-accuracy 维度: MDM ({100*r_mdm["frame_strict_acc"]:.0f}%) >> MDM+LeGO-CLIP ({100*r_lora["frame_strict_acc"]:.0f}%) > LeGO ({100*r_lego["frame_strict_acc"]:.0f}%)。
LeGO 系列模型靠转身来"作弊"实现方向区分, MDM+LeGO-CLIP 在一定程度上缓解了此问题。
逐帧来看: MDM 喂"向后走"时只有 {r_mdm["bwd_frame_pct_b"]:.0f}% 的移动帧真的向后(经常先往前走),
LeGO 对应比例 {r_lego["bwd_frame_pct_b"]:.0f}%, MDM+LeGO-CLIP 对应比例 {r_lora["bwd_frame_pct_b"]:.0f}%。""")

    # ---- 第4类: clockwise / counterclockwise ----
    cat = CATEGORIES[3]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第四类：顺/逆时针转身 (clockwise vs counterclockwise)
{'='*78}

【测试目的】给定含 "clockwise" 或 "counterclockwise" 的文本, 观察生成的人是否朝对应方向转身。

【范例文本】
  说"逆时针": "{cat['pairs'][0][0]}"
  说"顺时针": "{cat['pairs'][0][1]}"
  (共5组文本模板 × 20样本 = 每侧100个motion, 两侧共享同一组随机噪声)

【核心指标】m_yaw (身体朝向的旋转角速度, 单位 deg/s)
  计算方法: 逐帧计算身体朝向向量的有符号转角增量(用 atan2 保证 -180°~+180°),
           全程累加后除以时长。累加而非首末夹角: 转满一整圈首末夹角=0°(会被误判为没转),
           累加才能得到真正的总转角(例如360°)。
  符号约定: 正数(+) = 逆时针旋转(俯视, 即从头顶往下看, 身体向左转)
           负数(-) = 顺时针旋转(俯视, 即身体向右转)

【MDM 结果】
  说"逆时针"的100个motion, 平均旋转角速度 = {r_mdm["mean_a"]:+.2f} deg/s
    → 符号为{"正" if r_mdm["mean_a"]>0 else "负"}, {"正确(逆时针)" if r_mdm["mean_a"]>0 else "错误, 实际在顺时针转"}
  说"顺时针"的100个motion, 平均旋转角速度 = {r_mdm["mean_b"]:+.2f} deg/s
    → 符号为{"正" if r_mdm["mean_b"]>0 else "负"}, {"错误, 实际在逆时针转" if r_mdm["mean_b"]>0 else "正确(顺时针)"}
  两侧均值一正一负 (sign-flip): {"是" if r_mdm["sign_flip"] else "否 ← 两侧同号, 模型不管听什么词都朝同一方向转"}
  逐样本方向正确率 (correct): {100*r_mdm["sign_acc"]:.0f}% (瞎猜基线=50%)
    - A侧"说逆时针就逆时针"的比例: {100*r_mdm["sign_acc_a"]:.0f}%
    - B侧"说顺时针就顺时针"的比例: {100*r_mdm["sign_acc_b"]:.0f}%
  diff%: {100*r_mdm["diff_pct"]:.0f}%

【LeGO 结果】
  说"逆时针"的100个motion, 平均旋转角速度 = {r_lego["mean_a"]:+.2f} deg/s
    → 符号为{"正" if r_lego["mean_a"]>0 else "负"}, {"正确(逆时针)" if r_lego["mean_a"]>0 else "错误, 实际在顺时针转"}
  说"顺时针"的100个motion, 平均旋转角速度 = {r_lego["mean_b"]:+.2f} deg/s
    → 符号为{"正" if r_lego["mean_b"]>0 else "负"}, {"错误, 实际在逆时针转" if r_lego["mean_b"]>0 else "正确(顺时针)"}
  两侧均值一正一负 (sign-flip): {"是" if r_lego["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lego["sign_acc"]:.0f}%
    - A侧"说逆时针就逆时针"的比例: {100*r_lego["sign_acc_a"]:.0f}%
    - B侧"说顺时针就顺时针"的比例: {100*r_lego["sign_acc_b"]:.0f}%
  diff%: {100*r_lego["diff_pct"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"逆时针"的100个motion, 平均旋转角速度 = {r_lora["mean_a"]:+.2f} deg/s
    → 符号为{"正" if r_lora["mean_a"]>0 else "负"}, {"正确(逆时针)" if r_lora["mean_a"]>0 else "错误, 实际在顺时针转"}
  说"顺时针"的100个motion, 平均旋转角速度 = {r_lora["mean_b"]:+.2f} deg/s
    → 符号为{"正" if r_lora["mean_b"]>0 else "负"}, {"错误, 实际在逆时针转" if r_lora["mean_b"]>0 else "正确(顺时针)"}
  两侧均值一正一负 (sign-flip): {"是" if r_lora["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lora["sign_acc"]:.0f}%
    - A侧"说逆时针就逆时针"的比例: {100*r_lora["sign_acc_a"]:.0f}%
    - B侧"说顺时针就顺时针"的比例: {100*r_lora["sign_acc_b"]:.0f}%
  diff%: {100*r_lora["diff_pct"]:.0f}%

【结论】MDM+LeGO-CLIP (correct={100*r_lora["sign_acc"]:.0f}%) >> LeGO ({100*r_lego["sign_acc"]:.0f}%) > MDM ({100*r_mdm["sign_acc"]:.0f}%)。
这是三模型间差异最大的类别。MDM 两侧同号(sign-flip=否), 完全不理解 "counter-" 前缀, 只会顺时针转。
LeGO 实现了均值 sign-flip 但逐样本 correct 仅 60%(接近瞎猜)。
MDM+LeGO-CLIP correct=84%, 从"几乎瞎猜"跃升到"基本可靠", 且旋转速率远大于其他两模型。
预训练 LoRA CLIP 权重中似乎保留了更丰富的旋转方向语义。""")

    # ---- 第5类: left hand / right hand ----
    cat = CATEGORIES[4]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第五类：左手/右手举起 (left hand vs right hand)
{'='*78}

【测试目的】给定"边走边举左手"或"边走边举右手"的文本, 观察生成的人举的是否正确的那只手。

【范例文本】
  说"举左手": "{cat['pairs'][0][0]}"
  说"举右手": "{cat['pairs'][0][1]}"
  (共5组文本模板 × 20样本 = 每侧100个motion, 两侧共享同一组随机噪声)

【核心指标】m_hand (左右手腕高度差, 单位 m)
  计算方法: 全程(196帧 × 100个样本)中, 每一帧计算「左手腕高度 - 右手腕高度」,
           然后对所有帧、所有样本取平均值。
  含义: 正数(+) = 平均而言左手腕高于右手腕 → 举的是左手
        负数(-) = 平均而言右手腕高于左手腕 → 举的是右手
  为什么用差值而非单侧高度: 差值自动消掉"整体举手幅度"这个共同因素,
  只留下"举的是哪只手"的信号; 且对骨架尺寸不敏感。

【MDM 结果】
  说"举左手"的100个motion, 平均高度差 (左手-右手) = {r_mdm["mean_a"]:+.3f} m
    → 符号为{"正" if r_mdm["mean_a"]>0 else "负"}, {"正确, 左手确实更高" if r_mdm["mean_a"]>0 else "错误, 实际右手更高"}
  说"举右手"的100个motion, 平均高度差 (左手-右手) = {r_mdm["mean_b"]:+.3f} m
    → 符号为{"正" if r_mdm["mean_b"]>0 else "负"}, {"错误, 左手更高" if r_mdm["mean_b"]>0 else "正确, 右手确实更高"}
  两侧均值一正一负 (sign-flip): {"是" if r_mdm["sign_flip"] else "否 ← 两侧同号, 说明不管文本说举哪只手, 模型都举同一侧"}
  逐样本方向正确率 (correct): {100*r_mdm["sign_acc"]:.0f}% (瞎猜基线=50%)
    - A侧"说举左手真举左手"的比例: {100*r_mdm["sign_acc_a"]:.0f}%
    - B侧"说举右手真举右手"的比例: {100*r_mdm["sign_acc_b"]:.0f}%
  diff%: {100*r_mdm["diff_pct"]:.0f}%

【LeGO 结果】
  说"举左手"的100个motion, 平均高度差 (左手-右手) = {r_lego["mean_a"]:+.3f} m
    → 符号为{"正" if r_lego["mean_a"]>0 else "负"}, {"正确, 左手确实更高" if r_lego["mean_a"]>0 else "错误, 实际右手更高"}
  说"举右手"的100个motion, 平均高度差 (左手-右手) = {r_lego["mean_b"]:+.3f} m
    → 符号为{"正" if r_lego["mean_b"]>0 else "负"}, {"错误, 左手更高" if r_lego["mean_b"]>0 else "正确, 右手确实更高"}
  两侧均值一正一负 (sign-flip): {"是" if r_lego["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lego["sign_acc"]:.0f}%
    - A侧"说举左手真举左手"的比例: {100*r_lego["sign_acc_a"]:.0f}%
    - B侧"说举右手真举右手"的比例: {100*r_lego["sign_acc_b"]:.0f}%
  diff%: {100*r_lego["diff_pct"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"举左手"的100个motion, 平均高度差 (左手-右手) = {r_lora["mean_a"]:+.3f} m
    → 符号为{"正" if r_lora["mean_a"]>0 else "负"}, {"正确, 左手确实更高" if r_lora["mean_a"]>0 else "错误, 实际右手更高"}
  说"举右手"的100个motion, 平均高度差 (左手-右手) = {r_lora["mean_b"]:+.3f} m
    → 符号为{"正" if r_lora["mean_b"]>0 else "负"}, {"错误, 左手更高" if r_lora["mean_b"]>0 else "正确, 右手确实更高"}
  两侧均值一正一负 (sign-flip): {"是" if r_lora["sign_flip"] else "否"}
  逐样本方向正确率 (correct): {100*r_lora["sign_acc"]:.0f}%
    - A侧"说举左手真举左手"的比例: {100*r_lora["sign_acc_a"]:.0f}%
    - B侧"说举右手真举右手"的比例: {100*r_lora["sign_acc_b"]:.0f}%
  diff%: {100*r_lora["diff_pct"]:.0f}%

【结论】MDM+LeGO-CLIP ({100*r_lora["sign_acc"]:.0f}%) ≈ LeGO ({100*r_lego["sign_acc"]:.0f}%) >> MDM ({100*r_mdm["sign_acc"]:.0f}%)。
两个 LoRA 模型都近乎完美地区分了左右手。MDM 的 A侧均值 = {r_mdm["mean_a"]:+.3f} m ({'正' if r_mdm['mean_a']>0 else '负'}值),
说明说"举左手"时它仍然在举右手 (A侧 correct 仅 {100*r_mdm["sign_acc_a"]:.0f}%)。
MDM+LeGO-CLIP 的 B侧高度差 ({r_lora['mean_b']:+.3f} m) 幅度最大, 右手举得更明显。
LoRA 微调后的 CLIP 准确编码了 "left arm" 和 "right arm" 的语义差异。""")

    # ---- 第6类: walk / run ----
    cat = CATEGORIES[5]
    rpt = cat['report_window']
    r_mdm = summary[('MDM', cat['name'], rpt)]
    r_lego = summary[('LeGO', cat['name'], rpt)]
    r_lora = summary[('MDM+LeGO-CLIP', cat['name'], rpt)]
    print(f"""
{'='*78}
第六类：走/跑速度 (run vs walk)
{'='*78}

【测试目的】给定含 "run" 或 "walk" 的文本, 观察生成的人是否跑得比走得更快。

【范例文本】
  说"跑": "{cat['pairs'][0][0]}"
  说"走": "{cat['pairs'][0][1]}"
  (共5组文本模板 × 20样本 = 每侧100个motion, 两侧共享同一组随机噪声)

【核心指标】m_speed (平均速度, 单位 m/s)
  计算方法: 同第二类, 排除静止帧, 只在移动帧上统计: 总路径 ÷ (移动帧数/20fps)。

【MDM 结果】
  说"跑"的100个motion, 平均速度(仅移动帧) = {r_mdm["mean_a"]:.3f} m/s
  说"走"的100个motion, 平均速度(仅移动帧) = {r_mdm["mean_b"]:.3f} m/s
  跑/走速度比: {r_mdm["ratio"]:.2f} 倍
  diff (跑均值 - 走均值): {r_mdm["diff"]:.3f} m/s
  diff% (换"run"→"walk"后速度变慢的配对比例): {100*r_mdm["diff_pct"]:.0f}% (瞎猜基线=50%)

【LeGO 结果】
  说"跑"的100个motion, 平均速度(仅移动帧) = {r_lego["mean_a"]:.3f} m/s
  说"走"的100个motion, 平均速度(仅移动帧) = {r_lego["mean_b"]:.3f} m/s
  跑/走速度比: {r_lego["ratio"]:.2f} 倍
  diff: {r_lego["diff"]:.3f} m/s
  diff%: {100*r_lego["diff_pct"]:.0f}%

【MDM+LeGO-CLIP 结果】
  说"跑"的100个motion, 平均速度(仅移动帧) = {r_lora["mean_a"]:.3f} m/s
  说"走"的100个motion, 平均速度(仅移动帧) = {r_lora["mean_b"]:.3f} m/s
  跑/走速度比: {r_lora["ratio"]:.2f} 倍
  diff: {r_lora["diff"]:.3f} m/s
  diff%: {100*r_lora["diff_pct"]:.0f}%

【结论】这是所有 LoRA 模型共同失败的类别。MDM diff%={100*r_mdm["diff_pct"]:.0f}% >> MDM+LeGO-CLIP diff%={100*r_lora["diff_pct"]:.0f}% > LeGO diff%={100*r_lego["diff_pct"]:.0f}%。
MDM 清楚地区分了跑和走 (速度比 {r_mdm['ratio']:.2f}x), LeGO 完全学反 (速度比 {r_lego['ratio']:.2f}x, 说"走"反而更快),
MDM+LeGO-CLIP 有微弱改善 (速度比 {r_lora['ratio']:.2f}x, diff 从 LeGO 的 {r_lego['diff']:.3f} 回升至 {r_lora['diff']:+.3f}) 但 diff% 仍接近瞎猜水平。
原因分析: HumanML3D 中 walk(94.46%) vs run(12.81%) 极度不平衡;
LoRA 的低秩约束偏向于微调已有的语义维度(如速度副词)而难以建立新类别对比。
(注意: 两个 LoRA 模型在 slowly/quickly 上表现优异, 说明它们能理解速度副词, 但不理解动作类别词。)""")


if __name__ == '__main__':
    if CALIBRATE_ONLY:
        calibrate_on_gt()
    else:
        main()
