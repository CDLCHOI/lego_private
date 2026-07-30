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
]

# 每类: 主判据 key, 单位, A/B 语义标签, 汇报窗口, 5 组文本对(A = 期望主判据更大的一侧)
CATEGORIES = [
    {
        'name': 'left_right',
        'metric': 'm_lat_cos',
        'unit': 'cos',
        'angle_key': 'disp_angle',       # 展示用角度(度)
        'angle_desc': '位移方向偏离初始朝向的角度, 值域±180, +=偏左 -=偏右, ±90=正侧向',
        'constraint': None,
        'label_a': 'left', 'label_b': 'right',
        'signed': True,
        'filter_no_move': True,          # 余弦要除以位移模长, 静止样本是纯噪声, 必须剔除
        'report_window': '6s',       # 净位移会被 U 型回转抵消, 用短窗口(见 md §4.1)
        'desc': 'direction of travel relative to the initial facing direction (cosine, +1 = straight left)',
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
        # 文本明确说了 "facing forward", 故额外要求朝向保持: 首末朝向夹角 < 90 度。
        # m_align 只管"走向 vs 朝向"的相对关系, 管不了"朝向有没有留在原始前方"。
        'constraint': {'key': 'heading_dev', 'max': 90.0,
                       'desc': '朝向偏离初始方向 < 90度 (即文本要求的 "facing forward" 被保持)'},
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
    """slowly/quickly: 每秒位移大小(路径长度/时长), 恒正"""
    dt = cache['dt']
    step = np.linalg.norm(cache['disp'], axis=-1)
    moving = step * FPS > MOVE_EPS
    return {
        'm_speed': float(cache['path']) / dt,
        # 辅助: 只在运动帧上平均, 区分"走得慢"与"走得少"
        'speed_loco': float((step[moving] * FPS).mean()) if moving.any() else 0.0,
        'cadence': count_steps(cache['contact'], s, e) / dt,
        'energy': motion_energy(joints, s, e),
    }


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
    proj = float(np.sum(cache['disp'] * cache['forward'][s:e - 1]))
    path = cache['path']
    m_align = proj / path if path > 1e-6 else 0.0

    # 朝向相对窗口起始帧偏离了多少度, 值域 [0,180]。
    # 注意必须用"首末朝向夹角"而不是累计转角 abs_yaw: 后者转满 360 度会得到 360,
    # 但实际朝向已回到原方向, 用它判"faces forward 是否保持"会误杀。
    f0, f1 = cache['forward'][s], cache['forward'][e - 1]
    heading_dev = float(np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1.0, 1.0))))

    return {
        # 主判据: 方向余弦 [-1,1]
        'm_align': m_align,
        # 展示用: 走向与朝向的夹角(度), 0 = 完全朝前走, 180 = 完全倒着走
        'align_angle': float(np.degrees(np.arccos(np.clip(m_align, -1.0, 1.0)))),
        # 辅助: 朝向偏离度, 文本说 "facing forward" 时应接近 0 -> 用于排除转身情形
        'heading_dev': heading_dev,
        # 辅助: 未归一化版本(m/s), 说明幅度
        'm_fwd': proj / dt,
        # 辅助: 用第 0 帧固定前轴(等价于 z 坐标差), 会被"转身后往前走"骗过, 仅作对照
        'fwd_disp_ref': float(np.dot(cache['p'][e - 1] - cache['p'][s], cache['forward'][s])) / dt,
        # 辅助: 累计转角绝对值(仅供参考, 判朝向保持请用 heading_dev)
        'abs_yaw': float(abs(np.degrees(cache['yaw_delta'][s:e - 1].sum()))),
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
    out.update(metric_facing_disp(joints, s, e, cache))
    out.update(metric_turn_rate(joints, s, e, cache))
    out.update(metric_hand_height(joints, s, e, cache))
    return out


# ==================== 汇总统计 (md §2) ====================
def summarize_pair(vals_a, vals_b, no_move_a=None, no_move_b=None, signed=True,
                   filter_no_move=True, angle_a=None, angle_b=None,
                   con_a=None, con_b=None, con_max=None):
    """对内两条文本的汇总: 均值 / gap / 倍率 / PPA / 绝对符号准确率 / 静止率

    vals_a, vals_b: (n,) 同一组配对噪声下 A / B 两条文本的主判据值

    filter_no_move: 归一化的方向判据(余弦)分母是位移模长/里程, 静止样本会得到纯噪声,
                    必须整体剔除(均值和准确率都剔)。而 speed 类"静止"本身就是慢的信息、
                    cw_ccw 类"原地转身"是合法实现, 这两类不能剔。

    两个准确率互补, 缺一不可:
      PPA      配对差值符号正确的比例 —— 衡量"相对区分度"。但它对绝对方向不敏感:
               若某模型对 ccw 给 -0.4、对 cw 给 -10.8(两侧都在顺时针转), 差值方向仍"对",
               PPA 会虚高到 92%, 掩盖了"模型根本不懂 counter 前缀"这一事实。
      sign_acc A 侧 m>0 且 B 侧 m<0 的比例 —— 衡量"绝对方向是否正确", 正是 PPA 漏掉的部分。
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
    diff = a - b
    ppa = float((diff > 0).mean())

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

    # 严格准确率: 符号正确 **且** 满足附加约束(如 fwd_bwd 要求朝向留在原始前方)。
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
        'gap': float(a.mean() - b.mean()),
        'ratio': ratio,
        'ppa': ppa,
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
指标含义 (看不懂下面的数字时回来查这里)
------------------------------------------------------------------------------
  A侧 / B侧   每类有一对反义文本。A侧是"期望主判据更大"的那条, B侧是反义的那条。
              例: left/right 类 A侧="left"、B侧="right"; 举手类 A侧="left hand"。

  主判据      每类一个标量, 在该类 100 个配对样本(5个文本模板 x 20个样本)上取均值。
              每类的定义、单位、符号约定见该类标题下的说明。

  [对]/[错]   该侧均值的符号是否符合期望。期望是: A侧为正, B侧为负。
              speed 类数值恒正、无符号可言, 故不标注。

  gap         = A侧均值 - B侧均值。越大 = 换个关键词动作变化越大 = 模型对关键词越敏感。
              带单位, 可直接理解为"两个关键词造成的物理差异有多大"。

  PPA         = 100 组配对中"A侧数值 > B侧数值"的比例。瞎猜 = 50%。
              衡量【相对区分度】。**局限**: 两侧同号时 PPA 仍可能很高。
              例: 某模型对 ccw 给 -0.4、对 cw 给 -10.8(两次都在顺时针转, 绝对方向全错),
              但 -0.4 > -10.8 成立, PPA 仍高达 92%。所以 PPA 必须配合 dir-acc 一起看。

  dir-acc     = 绝对方向准确率 = (A侧数值为正的比例 + B侧数值为负的比例) / 2。瞎猜 = 50%。
              括号内 (A侧% / B侧%) 是两侧各自的准确率。
              衡量【绝对方向对不对】, 正是 PPA 漏掉的部分。**这是最该看的指标。**

  sign flip   A侧均值为正 且 B侧均值为负, 即干净的符号翻转 -> 模型真正区分了这对关键词。
              两侧同号 = 不管文本说哪个词, 模型都朝同一个方向做 = 没有区分。

  angle       主判据换算成角度后的直观表述(仅 left/right 和 fwd/bwd 两类有)。

  一眼看好坏: 先看 A侧(斜杠左边)的符号对不对, 再比 dir-acc, 最后看 gap 的大小。
------------------------------------------------------------------------------"""


def verdict_text(r, signed):
    """给该模型在该类上的一句话判定

    注意不能只看均值的符号翻转: 均值翻转只说明"平均而言两侧朝相反方向", 但逐样本可能
    有很大比例是错的(例: 均值 +0.29/-0.30 看似完美翻转, dir-acc 却只有 65%)。
    所以判定必须结合 dir-acc。
    """
    if not signed:
        return f">> A侧是B侧的 {r['ratio']:.2f} 倍" if not np.isnan(r['ratio']) else ''

    da = r['sign_acc']
    if not r['sign_flip']:
        same = '都为正' if r['mean_a'] > 0 and r['mean_b'] > 0 else '都为负'
        return f'>> 两侧同号({same}) -> 不管文本说哪个词都朝同一方向做, 未真正区分关键词'
    sa = r.get('strict_acc', float('nan'))
    if not np.isnan(sa) and da - sa > 0.15:
        return (f'>> 方向 dir-acc {100*da:.0f}% 看似很好, 但加上附加约束后只剩 {100*sa:.0f}% '
                f'-> 大量样本靠"违反约束"达成方向, 不能算真正遵循文本')
    if da >= 0.95:
        return f'>> 符号翻转 且 逐样本 dir-acc {100*da:.0f}% -> 稳定地区分了这对关键词'
    if da >= 0.80:
        return f'>> 符号翻转, 逐样本 dir-acc {100*da:.0f}% -> 基本区分了这对关键词, 有少量样本方向错'
    if da >= 0.70:
        return f'>> 均值符号翻转, 但逐样本 dir-acc 仅 {100*da:.0f}% -> 区分不稳定, 相当一部分样本方向是错的'
    return (f'>> 均值虽符号翻转, 但逐样本 dir-acc 仅 {100*da:.0f}% (瞎猜是50%) '
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
        L.append(f"            -> 严格准确率 = 方向符号正确 **且** 满足该约束的比例")
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
               f' | dir-acc {100*r["sign_acc"]:.0f}% (A侧 {100*r["sign_acc_a"]:.0f}% / B侧 {100*r["sign_acc_b"]:.0f}%)')
        L.append(f'  {"":5s}  gap {r["gap"]:.3f} {unit} | PPA {100*r["ppa"]:.0f}%{acc}'
                 f' | 有效样本 {r["n_valid"]}/{N_SAMPLES*len(cat["pairs"])}')
        if not np.isnan(r['strict_acc']):
            L.append(f'  {"":5s}  违反约束比例: A侧 {100*r["violate_a"]:.0f}% / B侧 {100*r["violate_b"]:.0f}%'
                     f'  ->  严格准确率 {100*r["strict_acc"]:.0f}%'
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
                             'con_a': [], 'con_b': []} for w in EVAL_WINDOWS}
            skate_all = []

            for tid, (text_a, text_b) in enumerate(cat['pairs']):
                # 对内两条文本 + 两个模型共用同一 seed -> 完全配对。
                # 不能用 hash(str): Python3 的字符串 hash 受 PYTHONHASHSEED 随机化, 跨进程不稳定。
                seed = BASE_SEED + cat_idx * 100 + tid
                joints = {}
                for side, text in (('a', text_a), ('b', text_b)):
                    joints[side] = get_joints(diffusion, model_name, cat['name'], tid, side,
                                              text, seed, mean, std)
                    skate_all.append(skating(joints[side]))

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
                            sample_rows.append(dict(
                                model=model_name, category=cat['name'], template_id=tid,
                                side=side, keyword=cat['label_a'] if side == 'a' else cat['label_b'],
                                text=text, sample_idx=i, window=wname,
                                main_metric=cat['metric'], main_value=m[cat['metric']], **m))

                print(f'  [{cat["name"]}] template {tid} done')

            skate_mean = float(np.concatenate(skate_all).mean())
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
                r['skating'] = skate_mean
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
                'mean_b', 'std_b', 'angle_a', 'angle_b', 'gap', 'ratio', 'ppa', 'sign_acc',
                'sign_acc_a', 'sign_acc_b', 'strict_acc', 'strict_acc_a', 'strict_acc_b',
                'violate_a', 'violate_b', 'n_valid', 'no_move_rate', 'sign_flip', 'skating']
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
        sign_hint = (f'斜杠左 = A侧 "{cat["label_a"]}" (期望 >0), '
                     f'斜杠右 = B侧 "{cat["label_b"]}" (期望 <0)') if cat['signed'] else \
                    (f'斜杠左 = A侧 "{cat["label_a"]}", 斜杠右 = B侧 "{cat["label_b"]}" (期望 左 > 右)')
        print(f"\n{cat['name']}   主判据 {cat['metric']} (单位 {cat['unit']})")
        print(f"  {sign_hint}")
        for wname, _, _ in EVAL_WINDOWS:
            for mname, _, _ in MODELS:
                r = summary.get((mname, cat['name'], wname))
                if not r:
                    continue
                da = '' if np.isnan(r['sign_acc']) else f'  dir-acc {100*r["sign_acc"]:3.0f}%'
                flip = '  <- 符号翻转' if r['sign_flip'] else ''
                print(f'  {wname:5s} {mname:5s}: {r["mean_a"]:+7.2f} / {r["mean_b"]:+7.2f}'
                      f'  gap {r["gap"]:6.2f}  PPA {100*r["ppa"]:3.0f}%{da}{flip}')

    print('\n\n' + '=' * 78)
    print('质量守护: foot skating ratio (越低越好, 复用 utils/metrics.py:480 的标准实现)')
    print('=' * 78)
    for cat in CATEGORIES:
        line = f"  {cat['name']:12s}"
        for mname, _, _ in MODELS:
            r = summary.get((mname, cat['name'], cat['report_window']))
            if r:
                line += f'   {mname} {r["skating"]:.4f}'
        print(line)

    print(f'\n已写出 {OUT_SAMPLES_CSV} (逐样本) 和 {OUT_SUMMARY_CSV} (汇总)')


if __name__ == '__main__':
    if CALIBRATE_ONLY:
        calibrate_on_gt()
    else:
        main()
