import pandas as pd
import numpy as np
from typing import List, Tuple
import random
import math
import itertools
import datetime
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def pack_products_into_restrictions(items: List[List[float]], box: Tuple[float, float, float],
                                    record_over_size=False, try_rotate=False) -> Tuple[
    float, list, list, list]:
    # 保留浮点精度
    box_d, box_w, box_h = box
    placed_items = []
    placed_positions = []
    over_size_items = []

    # 空间管理改用几何碰撞检测
    occupied_regions = []  # 格式: [(x1, y1, z1, x2, y2, z2)]

    def check_collision(new_region):
        ''' AABB碰撞检测 '''
        nx1, ny1, nz1, nx2, ny2, nz2 = new_region
        for (x1, y1, z1, x2, y2, z2) in occupied_regions:
            if (nx1 < x2 and nx2 > x1 and
                    ny1 < y2 and ny2 > y1 and
                    nz1 < z2 and nz2 > z1):
                return True
        return False

    # 生成候选点（基于现有物品边缘）
    candidate_points = [(0.0, 0.0, 0.0)]

    for idx, item in enumerate(items):
        item_d, item_w, item_h = item
        can_place = False
        if try_rotate:
            rotations = generate_rotations(item)
            for rot in rotations:
                rot_d, rot_w, rot_h = rot
                if rot_d <= box_d and rot_w <= box_w and rot_h <= box_h:
                    item = rot
                    can_place = True
                    break
        else:
            can_place = item_d <= box_d and item_w <= box_w and item_h <= box_h

        if not can_place:
            if record_over_size:
                over_size_items.append(item)
            continue

        placed = False
        # 按候选点排序：优先低高度、小宽度、小深度
        for point in sorted(candidate_points, key=lambda p: (p[2], p[1], p[0])):
            x, y, z = point
            # 检查是否越界
            if (x + item[0] > box_d or
                    y + item[1] > box_w or
                    z + item[2] > box_h):
                continue

            # 定义新物品占据区域
            new_region = (x, y, z, x + item[0], y + item[1], z + item[2])

            # 碰撞检测
            if not check_collision(new_region):
                # 记录占位
                occupied_regions.append(new_region)
                placed_items.append((idx, item))
                placed_positions.append((x, y, z))

                # 生成新候选点（仅在三个轴向扩展）
                candidate_points.append((x + item[0], y, z))
                candidate_points.append((x, y + item[1], z))
                candidate_points.append((x, y, z + item[2]))

                # 移除已用候选点
                try:
                    candidate_points.remove(point)
                except ValueError:
                    pass
                placed = True
                break

        if not placed:
            continue

    # 计算空间利用率
    total_volume = box_d * box_w * box_h
    used_volume = sum(d * w * h for _, (d, w, h) in placed_items)
    space_ratio = used_volume / total_volume if total_volume > 0 else 0

    return space_ratio, placed_items, placed_positions, over_size_items

def generate_rotations(item):
    """生成浮点数的旋转方向"""
    return list({tuple(perm) for perm in itertools.permutations(item)})

def neighborhood_operator(current_items, box):
    new_items = [list(item) for item in current_items]

    # 操作1：交换物品顺序
    # if random.random() < 0.3:
    #     i, j = random.sample(range(len(new_items)), 2)
    #     new_items[i], new_items[j] = new_items[j], new_items[i]
    if random.random() < 0.3:
        if len(new_items) >= 2:  # 添加长度检查
            i, j = random.sample(range(len(new_items)), 2)
            new_items[i], new_items[j] = new_items[j], new_items[i]
        # 否则跳过交换操作
    # 操作2：旋转物品
    elif random.random() < 0.6 and new_items:
        idx = random.randint(0, len(new_items) - 1)
        original = new_items[idx]
        best_rot = original
        best_fit = 0

        for rot in generate_rotations(original):
            # 评估旋转方向适配度
            fit_score = sum(r <= b for r, b in zip(rot, box))
            if fit_score > best_fit:
                best_rot = rot
                best_fit = fit_score
        new_items[idx] = best_rot

    # 操作3：随机移除并插入物品
    elif new_items:
        idx = random.randint(0, len(new_items) - 1)
        item = new_items.pop(idx)
        if new_items:
            insert_pos = random.randint(0, len(new_items))
        else:
            insert_pos = 0

        new_items.insert(insert_pos, item)

    return new_items

def simulated_annealing(items, box,
                        t_initial=1000, t_min=1, alpha=0.95,
                        max_iter=500):
    # 预处理：过滤过大物品
    valid_items = []
    for item in items:
        if all(dim <= box[i] for i, dim in enumerate(item)):
            valid_items.append(item)

    best_ratio = 0
    best_items = []
    all_history = []

    # 多次重启
    num_restarts = 20
    for _ in range(num_restarts):
        current_items = sorted(valid_items,
                               key=lambda x: (x[0] * x[1] * x[2], max(x)),
                               reverse=True)
        random.shuffle(current_items)

        t = t_initial
        history = []

        for iter in range(max_iter):
            # 自适应调整温度下降速率
            if iter % 100 == 0 and iter > 0:
                alpha = 0.95 + random.uniform(-0.05, 0.05)

            # 生成新解
            new_items = neighborhood_operator(current_items, box)
            new_ratio, _, _, _ = pack_products_into_restrictions(new_items, box)

            # 退火接受准则
            delta = new_ratio - best_ratio
            if delta > 0 or math.exp(delta / t) > random.random():
                current_items = new_items
                if new_ratio > best_ratio:
                    best_items = current_items.copy()
                    best_ratio = new_ratio

            # 降温
            t *= alpha
            history.append(best_ratio)

            if iter % 50 == 0:
                print(f"Iter {iter}: Temp {t:.2f} Best {best_ratio:.2%}")

        all_history.extend(history)

    return best_ratio, best_items, all_history

# 可视化函数（支持浮点数）
def visualize_float_packing(box, items, positions, box_index):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制容器
    for z in [0, box[2]]:
        ax.plot([0, box[0], box[0], 0, 0],
                [0, 0, box[1], box[1], 0],
                [z] * 5, color='blue', alpha=0.3)

    # 绘制物品
    colors = plt.cm.tab20(np.linspace(0, 1, len(items)))
    for (pos, item), color in zip(zip(positions, items), colors):
        x, y, z = pos
        dx, dy, dz = item

        # 绘制三维长方体
        vertices = [
            [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
            [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.8))

    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_zlim(0, box[2])
    ax.set_xlabel('Depth')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    plt.title(f"Packing Visualization for Box {box_index}")

    # 添加物品尺寸信息
    item_info = "\n".join([f"Item {i+1}: {item}" for i, item in enumerate(items)])
    ax.text2D(1.05, 0.5, item_info, transform=ax.transAxes, fontsize=10,
               verticalalignment='center')

    plt.show()

if __name__=='__main__':
    ###导入数据
    excel = pd.ExcelFile('D:/qq文件/2025新生杯题目和附件/B题附件/data.xlsx')

    df = excel.parse('订单信息')

    new_df = df[['订单序号', 'TL', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Num']]

    # 设置列名
    new_df.columns = ['订单序号', 'TL', 'L', 'W', 'H', 'num']

    # 按订单序号分组并转换为列表
    result = new_df.groupby('订单序号')[['TL', 'L', 'W', 'H', 'num']].apply(lambda x: x.to_dict('records')).to_dict()

    # 用于保存结果的列表
    results = []

    for order_number, order_data in result.items():
        lwh_list = []
        for record in order_data:
            # 提取 L、W、H 数据并组合成一个数组
            lwh = [record['L'], record['W'], record['H']]
            num = record['num']
            tl = record['TL']
            for i in range(num):
                lwh_list.append(lwh)
        if tl == 0:
            boxes = (31, 19.4, 15), (34.5, 24.5, 18.5), (41, 26.0, 25.0), (48.0, 30.5, 25)
            lwh_list.append([15,11,2.5])
            lwh_list.append([15,11,2.5])

        else:
            boxes = (22, 15, 12), (29, 19.5, 17), (30, 24, 15.5), (36, 20, 21), (37.0, 30, 17), (36.0, 30, 25)
        items = lwh_list
        best_ratio = 0
        best_box = None
        for i, box in enumerate(boxes):
            print(f"Processing Box {i + 1}...")
            start = datetime.datetime.now()
            space_ratio, best_items, history = simulated_annealing(items, box)
            space_ratio_, placed_items, positions, over_size_items = pack_products_into_restrictions(best_items, box,
                                                                                                    record_over_size=True,
                                                                                                    try_rotate=True)

            print(f"优化耗时: {datetime.datetime.now() - start}")
            print(f"空间利用率: {space_ratio:.2%}")
            if over_size_items:
                print(f"超出尺寸的物品: {over_size_items}")

            visualize_float_packing(box, [item for _, item in placed_items], positions, i + 1)

            plt.plot(history)
            plt.title(f"Optimization Process for Box {i + 1}")
            plt.xlabel("Iteration")
            plt.ylabel("Space Ratio")
            # plt.show()qqqq

            # 检查是否所有物品都被放置且空间利用率更高
            if len(best_items)==len(items) and space_ratio > best_ratio:
                best_ratio = space_ratio
                best_box = box

        # 保存结果
        results.append({
            '订单序号': order_number,
            '最佳箱子类型': best_box,
            '最大空间利用率': best_ratio
        })

    # 创建 DataFrame
    result_df = pd.DataFrame(results)

    # 将结果保存到 Excel 文件
    result_df.to_excel('./packing_result4.xlsx', index=False)
