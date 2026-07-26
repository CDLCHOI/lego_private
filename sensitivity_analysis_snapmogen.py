import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch
import clip
import argparse
import glob
import ipdb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from models.LAMP.LAMP_minimal_text_net import Net
import json

def get_paired_text():

    if args.only_test_set:
        test_split_file = 'dataset/HumanML3D/test.txt'
        with open(test_split_file, 'r') as f:
            lines = f.readlines()
            test_names = [line.strip() for line in lines]
        text_files = [os.path.join('dataset/HumanML3D/texts', test_name + '.txt') for test_name in test_names]
    else:
        text_files = glob.glob(os.path.join('dataset/HumanML3D/texts', '*.txt'))
    print('num of text files: ', len(text_files))

    keyword_list = []
    paired_texts = []
    keywords =[
            ['clockwise', 'counterclockwise'],
            ['slowly', 'quickly'],
            ['left', 'right'],
            ['forward', 'backward'],
            ['walks', 'runs'],
            ['walking', 'running'],
            ]

    for text_file in text_files:
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip().split('#')
                caption = line_split[0]
                
                np.random.shuffle(keywords)
                for keyword in keywords: # 一个句子有多个关键词，每一次都会对所有关键词替换并构造pair
                    if keyword[0] in caption:
                        caption_2 = caption.replace(keyword[0], keyword[1])
                        paired_texts.append((caption, caption_2))
                        keyword_list.append(keyword[0])
                    elif keyword[1] in caption:
                        caption_2 = caption.replace(keyword[1], keyword[0])
                        paired_texts.append((caption, caption_2))
                        keyword_list.append(keyword[1])
        
        if len(paired_texts) >= args.max_samples:
            print('len(paired_texts) >= args.max_samples, break')
            break

    print('num of paired texts in HumanML3D: ', len(paired_texts))
    return paired_texts, keyword_list

def get_paired_text_snapomogen():
    
    caption_file = 'dataset/SnapMoGen/all_caption_clean.json'
    with open(caption_file, 'r') as f:
        caption_data = json.load(f)
    
    # print(f'num of entries in SnapMoGen: {len(caption_data)}')

    keyword_list = []
    paired_texts = []
    keywords =[
            ['clockwise', 'counterclockwise'],
            ['slowly', 'quickly'],
            ['left', 'right'],
            ['forward', 'backward'],
            ['walks', 'runs'],
            ['walking', 'running'],
            ]

    i = 0
    for entry_id, entry_data in caption_data.items():
        captions = entry_data.get('gpt', []) + entry_data.get('manual', [])
        
        for caption in captions:
            if 'The woman stands in a relaxed posture' in caption:
                a = 1
            # print(i, len(caption.split()))
            np.random.shuffle(keywords)
            for keyword in keywords: # 一个句子有多个关键词，每一次都会对所有关键词替换并构造pair
                if keyword[0] in caption:
                    caption_2 = caption.replace(keyword[0], keyword[1])
                    paired_texts.append((caption, caption_2))
                    keyword_list.append(keyword[0])
                elif keyword[1] in caption:
                    caption_2 = caption.replace(keyword[1], keyword[0])
                    paired_texts.append((caption, caption_2))
                    keyword_list.append(keyword[1])
            break
        i += 1
        
        if len(paired_texts) >= args.max_samples:
            print('len(paired_texts) >= args.max_samples, break')
            break

    print('num of paired texts in SnapMoGen: ', len(paired_texts))
    return paired_texts, keyword_list

def calc_mean_similarity(paired_texts, clip_model, batch_size=32, max_samples=40000, is_lamp=False):
    mean_similarity = 0.0
    total_processed = 0

    # 提取所有需要处理的caption，最多处理max_samples个
    all_captions1 = []
    all_captions2 = []
    for i, (caption1, caption2) in enumerate(paired_texts):
        # if i >= max_samples:
        #     break
        all_captions1.append(caption1)
        all_captions2.append(caption2)
    
    total_pairs = len(all_captions1)
    
    sim_list = []
    emb1 = []
    emb2 = []

    # 分批次处理
    for i in range(0, total_pairs, batch_size):
        if i%1000 == 0:
            print(f'processed {i}/{total_pairs} pairs')
        # 获取当前批次的文本
        batch_captions1 = all_captions1[i:i+batch_size]
        batch_captions2 = all_captions2[i:i+batch_size]
        
        if is_lamp:
            batch_emb1 = clip_model(batch_captions1)
            batch_emb2 = clip_model(batch_captions2)
        else:
            # 批量tokenize
            batch_text1 = clip.tokenize(batch_captions1, truncate=True).cuda()
            batch_text2 = clip.tokenize(batch_captions2, truncate=True).cuda()
            
            # 批量提取embedding
            with torch.no_grad():
                batch_emb1 = clip_model.encode_text(batch_text1).float()
                batch_emb2 = clip_model.encode_text(batch_text2).float()
        
        # 归一化
        batch_emb1_norm = batch_emb1 / batch_emb1.norm(dim=1, keepdim=True)
        batch_emb2_norm = batch_emb2 / batch_emb2.norm(dim=1, keepdim=True)
            
        emb1.append(batch_emb1_norm.detach().cpu().numpy())
        emb2.append(batch_emb2_norm.detach().cpu().numpy())
        
        # 计算余弦相似度
        batch_similarity = torch.cosine_similarity(batch_emb1_norm, batch_emb2_norm, dim=-1)
        # sim_list.append(batch_similarity.mean().item())
        sim_list.extend(batch_similarity.tolist())
        
        # 累加到总相似度
        mean_similarity += batch_similarity.sum().item()
        
        # 更新已处理数量
        current_batch_size = len(batch_captions1)
        total_processed += current_batch_size
        
    
    # 计算平均相似度
    if total_processed > 0:
        mean_similarity /= total_processed

    emb1 = np.concatenate(emb1, axis=0)
    emb2 = np.concatenate(emb2, axis=0)

    emb = np.reshape(np.stack((emb1, emb2), axis=1), (-1, emb1.shape[1]))
    
    return mean_similarity, np.array(sim_list), emb

def calc_avg_text_length_and_draw(sim_array, paired_texts, sample_text_filename, save_path):
    assert len(sim_array) == len(paired_texts), f"sim_array length {len(sim_array)} != paired_texts length {len(paired_texts)}"
    texts_length = [0] * 10
    count = [0] * 10

    # 计算平均文本长度
    for i in range(sim_array.shape[0]):
        sim = sim_array[i, 1]
        idx = int(sim * 10)
        texts_length[idx] += len(paired_texts[i][0].strip().split())
        count[idx] += 1

    # 计算每个区间的平均文本长度
    for i in range(10):
        if count[i] > 0:
            texts_length[i] /= count[i]

    
    with open(sample_text_filename, 'a') as f:
        for i in range(10):
            f.write(f"Similarity {i/10:.1f}~{i/10+0.1:.1f} avg text length: {texts_length[i]:.2f}\n")

    ################ 画图

    # 定义横纵坐标数组
    x_array = np.linspace(0,0.9,10) + 0.05 # 柱状图中心x坐标
    # 自定义10个y轴数值，shape为(10,)
    y_array = texts_length

    fig, ax = plt.subplots(figsize=(5, 4))

    # 绘制柱状图
    bars = ax.bar(x_array, y_array, color='#2C64E2', width=0.08)

    # 在每个柱子上方标注数值
    i = 0
    for bar in bars:
        # 获取柱子的高度（即y值）
        height = bar.get_height()
        # 在柱子顶部居中位置添加文本标注
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x坐标（柱子中心）
            height + 0.2,                       # y坐标（柱子顶部上方0.2，避免重叠）
            f'{y_array[i]:.1f}',                        # 要显示的数值
            ha='center',                        # 水平居中
            va='bottom',                        # 垂直靠下
            fontsize=10,                        # 字体大小
            fontweight='bold'                   # 字体加粗
        )
        i += 1

    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框

    # 设置图表标题和坐标轴标签
    ax.set_xlabel('Similarity', fontsize=12)
    ax.set_ylabel('Average Text Length', fontsize=12)
    

    # 设置x轴刻度（确保和x_array一致）
    ax.set_xticks(x_array-0.05)

    # 设置y轴范围（留出顶部空间，避免数值标注超出图表）
    ax.set_ylim(0, max(y_array) + 2)

    # 添加网格线（增强可读性）
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 显示图表
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()
    plt.savefig(save_path)
    
    return texts_length



# def plot_similarity_histogram(sim_array, save_path="similarity_histogram.png", bins=80):
#     """
#     绘制原始CLIP和AdaCLIP的余弦相似度柱状图
    
#     Args:
#         sim_array: 形状为(N,2)的numpy数组，第一列是原始CLIP的相似度，第二列是AdaCLIP的相似度
#         save_path: 图像保存路径
#         bins: 直方图的柱子数量
#     """
#     # 提取原始CLIP和AdaCLIP的相似度数据
#     clip_similarities = sim_array[:, 0]
#     adaclip_similarities = sim_array[:, 1]
#     if sim_array.shape[1] == 3:
#         lamp_similarities = sim_array[:, 2]
    
#     # 设置直方图参数
#     range_min = 0.2
#     range_max = 1.0
#     bin_edges = np.linspace(range_min, range_max, bins + 1)
    
#     # 绘制直方图
#     plt.figure(figsize=(8, 4))
    
#     # 原始CLIP用蓝色
#     plt.hist(clip_similarities, bins=bin_edges, alpha=0.7, color='#73D4F7', label='Vanilla CLIP')
    
#     # AdaCLIP用红色
#     plt.hist(adaclip_similarities, bins=bin_edges, alpha=0.7, color='#FF7373', label='AdaCLIP')

#     # LAMP用绿色
#     plt.hist(lamp_similarities, bins=bin_edges, alpha=0.7, color='#73F773', label='LAMP')
    
#     # 设置坐标轴范围和标签
#     plt.xlim(range_min, range_max)
#     plt.xlabel('Cosine Similarity', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
    
#     # 添加标题、图例和网格线
#     plt.title('Cosine Similarity Distribution between Vanilla CLIP and AdaCLIP', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.3)
    
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved similarity histogram to {save_path}")

def plot_3d_similarity_histogram(sim_array, save_path="3d_similarity_histogram.png", bins=80):
    """
    绘制三维余弦相似度频数分布柱状图
    
    Args:
        sim_array: 形状为(N,3)的numpy数组，列依次对应原始CLIP、AdaCLIP、LAMP的相似度
        save_path: 图像保存路径
        bins: 直方图的柱子数量（每个维度）
    """
    # 确保输入包含三个模型的数据
    if sim_array.shape[1] != 3:
        raise ValueError("3D直方图需要包含三个模型的相似度数据（CLIP, AdaCLIP, LAMP）")
    
    # 提取三个模型的相似度数据
    clip_sim = sim_array[:, 0]
    adaclip_sim = sim_array[:, 1]
    lamp_sim = sim_array[:, 2]
    
    # 设置相似度范围
    range_min = 0.2
    range_max = 1.0
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算每个模型的直方图
    clip_counts, clip_edges = np.histogram(clip_sim, bins=bins, range=(range_min, range_max))
    adaclip_counts, adaclip_edges = np.histogram(adaclip_sim, bins=bins, range=(range_min, range_max))
    lamp_counts, lamp_edges = np.histogram(lamp_sim, bins=bins, range=(range_min, range_max))
    
    # 计算每个bin的中心位置
    clip_centers = (clip_edges[:-1] + clip_edges[1:]) / 2
    lamp_centers = (lamp_edges[:-1] + lamp_edges[1:]) / 2
    adaclip_centers = (adaclip_edges[:-1] + adaclip_edges[1:]) / 2
    
    # 设置每个模型在Z轴上的位置（用于区分）
    ytick_clip = 0
    ytick_lamp = 0.5
    ytick_adaclip = 1

    z_pos_clip = np.ones_like(clip_centers) * ytick_clip
    z_pos_lamp = np.ones_like(lamp_centers) * ytick_lamp
    z_pos_adaclip = np.ones_like(adaclip_centers) * ytick_adaclip
    
    
    bar_width = (range_max - range_min) / bins   
    # 绘制3D柱状图
    # Vanilla CLIP (蓝色)
    ax.bar(clip_centers, clip_counts, zs=z_pos_clip, zdir='y', 
           width=bar_width, color='#1D1DFF', alpha=0.8, label='Vanilla CLIP')
    
    
    # LAMP (绿色)
    ax.bar(lamp_centers, lamp_counts, zs=z_pos_lamp, zdir='y', 
           width=bar_width, color='#17C913', alpha=0.8, label='LAMP')
    
    # AdaCLIP (红色)
    ax.bar(adaclip_centers, adaclip_counts, zs=z_pos_adaclip, zdir='y', 
           width=bar_width, color='#FF3D02', alpha=0.8, label='AdaCLIP')
    
    
    # 设置坐标轴标签
    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Model', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Frequency', fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_ylim(0, 1.2)
    ax.set_zlim(0, 2700)
    
    
    # 设置Y轴刻度和标签（对应不同模型）
    ax.set_yticks([ytick_clip, ytick_lamp, ytick_adaclip])
    ax.set_yticklabels(['Vanilla CLIP', 'LAMP', 'AdaCLIP'])
    
    # 设置X轴范围
    ax.set_xlim(range_min, range_max)
    # 添加标题和图例
    # ax.set_title('3D Cosine Similarity Distribution', fontsize=14, pad=20)
    ax.legend(fontsize=10, loc='upper right')
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    # 保存图像
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D similarity histogram to {save_path}")

def plot_similarity_by_text_length(sim_array, paired_texts, save_path):
    """
    绘制横坐标为文本长度、纵坐标为平均相似度的折线图，包含CLIP（第0维）、LAMP（第1维）、AdaCLIP（第2维）三条折线
    
    Args:
        sim_array: 形状为(N,3)的numpy数组，第一列是CLIP的相似度，第二列是AdaCLIP的相似度，第三列是LAMP的相似度
        paired_texts: 形状为(N,2)的文本对列表
        save_path: 图像保存路径
    """
    assert len(sim_array) == len(paired_texts), f"sim_array length {len(sim_array)} != paired_texts length {len(paired_texts)}"
    
    # 收集每个文本长度对应的相似度
    length_to_similarities = {}
    
    for i in range(sim_array.shape[0]):
        # 计算文本长度（使用第一个文本的单词数）
        text_length = len(paired_texts[i][0].strip().split())
        
        # 获取三个模型的相似度
        clip_sim = sim_array[i, 0]
        adaclip_sim = sim_array[i, 1]
        lamp_sim = sim_array[i, 2]
        
        # 将相似度添加到对应文本长度的列表中
        if text_length not in length_to_similarities:
            length_to_similarities[text_length] = {
                'clip': [],
                'adaclip': [],
                'lamp': []
            }
        
        length_to_similarities[text_length]['clip'].append(clip_sim)
        length_to_similarities[text_length]['adaclip'].append(adaclip_sim)
        length_to_similarities[text_length]['lamp'].append(lamp_sim)
    
    sorted_items = sorted(length_to_similarities.items(), key=lambda x: x[0]) 
    sorted_length_to_similarities = dict(sorted_items)


    # 打印每个文本长度下，有多少个样本
    valid_text_lengths = []
    for text_length in sorted_length_to_similarities.keys():
        num = len(sorted_length_to_similarities[text_length]['clip'])
        print(f"Text length {text_length}: {num} samples")
        if num >= 10:
            valid_text_lengths.append(text_length)


    # 计算每个文本长度下的平均相似度
    avg_clip_sims = []
    avg_adaclip_sims = []
    avg_lamp_sims = []
    
    for length in valid_text_lengths:
        sims = length_to_similarities[length]
        avg_clip_sims.append(np.mean(sims['clip']))
        avg_adaclip_sims.append(np.mean(sims['adaclip']))
        avg_lamp_sims.append(np.mean(sims['lamp']))
    
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    
    # 绘制三条折线
    plt.plot(valid_text_lengths, avg_clip_sims, marker='o', linestyle='-', color='#73D4F7', label='CLIP', linewidth=2)
    plt.plot(valid_text_lengths, avg_lamp_sims, marker='s', linestyle='-', color='#73F773', label='LAMP', linewidth=2)
    plt.plot(valid_text_lengths, avg_adaclip_sims, marker='^', linestyle='-', color='#FF7373', label='AdaCLIP', linewidth=2)
    
    # 设置图表标题和坐标轴标签
    plt.xlabel('Text Length', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=10)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴范围
    plt.xlim(min(valid_text_lengths) - 1, max(valid_text_lengths) + 1)
    plt.ylim(0.3, 1.0)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved similarity by text length plot to {save_path}")









def test(clip_version, batch_size, max_samples):
    """
    合并LoRA权重到原始CLIP模型中
    
    Args:
        clip_version: CLIP模型版本，如'ViT-B/32'
        lora_checkpoint_path: 包含训练好的LoRA权重的检查点路径
        output_path: 合并后的模型保存路径
    """
    # 1. 加载原始CLIP模型
    print(f"Loading original CLIP model: {clip_version}")
    original_clip, _ = clip.load(clip_version, device='cpu', jit=False)
    original_clip = original_clip.cuda()
    
    # 2. AdaCLIP
    adaclip, _ = clip.load(clip_version, device='cpu', jit=False)
    merge_clip_key = torch.load('output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch/merged_clip.pth')
    missing_keys, unexpected_keys = adaclip.load_state_dict(merge_clip_key, strict=True)
    adaclip = adaclip.cuda()

    # 3. LAMP
    lamp = Net()
    lamp = lamp.cuda()
    
    paired_texts, keyword_list = get_paired_text()
    # paired_texts, keyword_list = get_paired_text_snapomogen()
    paired_texts2, keyword_list2 = get_paired_text_snapomogen()
    paired_texts.extend(paired_texts2)
    keyword_list.extend(keyword_list2)

    sim_clip, sim_clip_array, clip_emb = calc_mean_similarity(paired_texts, original_clip, batch_size, max_samples)
    sim_adaclip, sim_adaclip_array, adaclip_emb = calc_mean_similarity(paired_texts, adaclip, batch_size, max_samples)
    sim_lamp, sim_lamp_array, lamp_emb = calc_mean_similarity(paired_texts, lamp, batch_size, max_samples, is_lamp=True)

    print(f"Mean similarity original CLIP: {sim_clip:.4f}, std: {sim_clip_array.std():.4f}")
    print(f"Mean similarity AdaCLIP: {sim_adaclip:.4f}, std: {sim_adaclip_array.std():.4f}")
    print(f"Mean similarity LAMP: {sim_lamp:.4f}, std: {sim_lamp_array.std():.4f}")

    # 由于batch size整除问题，这里统一长度
    sim_array = np.stack((sim_clip_array, sim_adaclip_array, sim_lamp_array), axis=1)
    min_len = min(sim_array.shape[0], len(paired_texts))
    sim_array = sim_array[:min_len]
    paired_texts = paired_texts[:min_len]
    keyword_list = keyword_list[:min_len]


    sim_array_filename = f'sensitivity_sim_array_{args.max_samples}_{"test" if args.only_test_set else "all"}.txt'
    sample_text_filename = f'sensitivity_sample_text_{args.max_samples}_{"test" if args.only_test_set else "all"}.txt'
    count = [0] * 10 # 10个区间
    texts_for_save = [[] for i in range(10)]
    max_num_per_interval = 50

    # 按照相同次序打乱sim_array和paired_texts
    N = sim_array.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    shuffled_paired_texts = [paired_texts[i] for i in indices]
    shuffled_keyword_list = [keyword_list[i] for i in indices]
    shuffled_sim_array = sim_array[indices]

    # with open(sim_array_filename, 'w') as f:
    for i in range(shuffled_sim_array.shape[0]):
        # f.write(f"{shuffled_sim_array[i, 0]:.4f} {shuffled_sim_array[i, 2]:.4f} {shuffled_sim_array[i, 1]:.4f}\n")
        sim = shuffled_sim_array[i, 1]
        idx = int(sim * 10)
        if idx > 9:
            idx = 9
        if count[idx] <= max_num_per_interval: # 每个区间最大数量
            # ~ & sim_lip & sim_lamp & sim_adaclip & text
            texts_for_save[idx].append(f"~ & {shuffled_sim_array[i, 0]:.3f} & {shuffled_sim_array[i, 2]:.3f} & {shuffled_sim_array[i, 1]:.3f} & {shuffled_keyword_list[i]} & {shuffled_paired_texts[i][0]} " + '\\\\')
            count[idx] += 1
    # print(f'save {sim_array_filename}')


    # 保存样本文本
    with open(sample_text_filename, 'w') as f:
        for i in range(10):
            f.write(f"Similarity {i/10:.1f}~{i/10+0.1:.1f}:\n")
            for text in texts_for_save[i]:
                f.write(f"{text}\n")
            f.write("\n")
    print(f'save {sample_text_filename}')

    # 统计平均文本长度, 并画图
    # calc_avg_text_length_and_draw(sim_array, paired_texts, sample_text_filename, save_path=f"sensitivity_avg_text_length_{args.max_samples}_{'test' if args.only_test_set else 'all'}.pdf")


    # tSNE画图
    # visualize_tsne(paired_texts, original_clip, "Original CLIP Embedding Space", "tsne_original.png", clip_emb)
    # visualize_tsne(paired_texts, adaclip, "AdaCLIP Embedding Space", "tsne_adaclip.png", adaclip_emb)
    
    # 绘制相似度柱状图
    # plot_similarity_histogram(sim_array, save_path=f"sensitivity_histogram_{args.max_samples}_{'test' if args.only_test_set else 'all'}.pdf")

    # 绘制3D相似度柱状图
    plot_3d_similarity_histogram(sim_array, save_path=f"sensitivity_histogram_3d_{args.max_samples}_{'test' if args.only_test_set else 'all'}.png")

    # 绘制按文本长度统计的平均相似度折线图
    plot_similarity_by_text_length(sim_array, paired_texts, save_path=f"sensitivity_similarity_by_text_length_{args.max_samples}_{'test' if args.only_test_set else 'all'}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into original CLIP model")
    parser.add_argument("--clip_version", type=str, default="ViT-B/32", help="CLIP model version")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for similarity calculation")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max number of samples for similarity calculation")
    parser.add_argument('--only_test_set', default=False, action='store_true', help='Only test on test set')
    
    args = parser.parse_args()

    args.only_test_set = True
    
    test(args.clip_version, args.batch_size, args.max_samples)