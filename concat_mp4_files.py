import cv2
import numpy as np
import os

def resize_to_height(clip, target_height):
    """调整视频帧高度，保持宽高比"""
    h, w = clip.shape[:2]
    new_width = int(w * target_height / h)
    return cv2.resize(clip, (new_width, target_height), interpolation=cv2.INTER_AREA)

def concatenate_videos_h(video_paths, output_path, fps=30):
    """
    将多个视频沿水平方向拼接并保存
    :param video_paths: 视频路径列表（例如 4 个视频）
    :param output_path: 输出视频路径
    :param fps: 输出帧率（可自动检测）
    """
    caps = [cv2.VideoCapture(path) for path in video_paths]

    # 检查是否所有视频都成功打开
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_paths[i]}")

    # 获取每个视频的帧率（取第一个）
    if fps is None:
        fps = caps[0].get(cv2.CAP_PROP_FPS)

    # 读取第一帧以获取尺寸
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频的首帧")
        frames.append(frame)

    # 设定目标高度（例如统一为 480）
    target_height = 480
    resized_frames = [resize_to_height(frame, target_height) for frame in frames]
    
    # 计算总宽度（所有视频宽度之和）
    total_width = sum(f.shape[1] for f in resized_frames)
    
    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 兼容性好的编码
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  # 兼容性好的编码
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, target_height))

    # 回到开头（因为前面读了一帧）
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 开始逐帧读取并拼接
    while True:
        imgs = []
        all_has_frame = True

        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                all_has_frame = False
                break
            imgs.append(frame)

        if not all_has_frame or len(imgs) != len(caps):
            break

        # 调整每帧到目标高度并水平拼接
        resized_imgs = [resize_to_height(img, target_height) for img in imgs]
        concatenated = np.hstack(resized_imgs)  # 水平拼接

        # 写入输出视频
        out.write(concatenated)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    print(f"视频已保存至: {output_path}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    
    path = [
        '/home/deli/project/reward_mdm/output/0814_MDMCLIP_b128/visualization',
        '/home/deli/project/reward_mdm/output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/visualization',
        '/home/deli/project/momask-codes/visualize',
        '/home/deli/project/salad/visualize'
    ]

    out_path = '/home/deli/project/reward_mdm/visualize4'

    for i in range(600):
        mp4_name = f'{i:03d}.mp4'
        mp4_name_list = [f'{p}/{mp4_name}' for p in path]

        out_file = os.path.join(out_path, mp4_name)

        concatenate_videos_h(mp4_name_list, out_file, fps=20)
        