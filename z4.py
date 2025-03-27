import numpy as np
from typing import List, Tuple
import random
import math
import itertools
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def pack_products_into_restrictions(items: List[List[float]], box: Tuple[float, float, float]) -> Tuple[
    float, list, list]:
    """支持浮点数的装箱核心算法"""
    # 保留浮点精度
    box_d, box_w, box_h = box
    placed_items = []
    placed_positions = []

    # 空间管理改用几何碰撞检测
    occupied_regions = []  # 格式: [(x1, y1, z1, x2, y2, z2)]

    def check_collision(new_region):
        """AABB碰撞检测"""
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
        if item_d > box_d or item_w > box_w or item_h > box_h:
            continue

        placed = False
        # 按候选点排序：优先低高度、小宽度、小深度
        for point in sorted(candidate_points, key=lambda p: (p[2], p[1], p[0])):
            x, y, z = point
            # 检查是否越界
            if (x + item_d > box_d or
                    y + item_w > box_w or
                    z + item_h > box_h):
                continue

            # 定义新物品占据区域
            new_region = (x, y, z, x + item_d, y + item_w, z + item_h)

            # 碰撞检测
            if not check_collision(new_region):
                # 记录占位
                occupied_regions.append(new_region)
                placed_items.append((idx, item))
                placed_positions.append((x, y, z))

                # 生成新候选点（仅在三个轴向扩展）
                candidate_points.append((x + item_d, y, z))
                candidate_points.append((x, y + item_w, z))
                candidate_points.append((x, y, z + item_h))

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

    return space_ratio, placed_items, placed_positions

def generate_rotations(item):
    """生成浮点数的旋转方向"""
    return list({tuple(perm) for perm in itertools.permutations(item)})

def neighborhood_operator(current_items, box):
    new_items = [list(item) for item in current_items]

    # 操作1：交换物品顺序
    if random.random() < 0.5:
        i, j = random.sample(range(len(new_items)), 2)
        new_items[i], new_items[j] = new_items[j], new_items[i]

    # 操作2：旋转物品
    else:
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

    return new_items

def simulated_annealing(items, box,
                        t_initial=1000, t_min=1, alpha=0.95,
                        max_iter=500):
    # 预处理：过滤过大物品
    valid_items = []
    for item in items:
        if all(dim <= box[i] for i, dim in enumerate(item)):
            valid_items.append(item)

    current_items = sorted(valid_items,
                           key=lambda x: (x[0] * x[1] * x[2], max(x)),
                           reverse=True)
    best_items = current_items.copy()
    best_ratio, _, _ = pack_products_into_restrictions(current_items, box)

    t = t_initial
    history = []

    for iter in range(max_iter):
        # 生成新解
        new_items = neighborhood_operator(current_items, box)
        new_ratio, _, _ = pack_products_into_restrictions(new_items, box)

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

    return best_ratio, best_items, history

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
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 浮点测试数据
    boxes = [
        (31, 19.4, 15), (34.5, 24.5, 18.5), (41, 26.0, 25.0), (48.0, 30.5, 25)
    ]
    items = [
        [20, 5, 27],
        [25, 24, 8.5],
        [20, 5, 27],
        [18, 13, 11],
        [26.2, 19.3, 6.5],
        [23.5, 16.5, 4.5],
        [16, 15.5, 2],
        [19, 11.5, 2.5], [19, 11.5, 2.5],
        [24, 16.5, 3], [15, 11, 2.5], [15, 11, 2.5]
    ]



    for i, box in enumerate(boxes):
        print(f"Processing Box {i + 1}...")
        start = datetime.datetime.now()
        ratio, best_items, history = simulated_annealing(items, box)
        _, _, positions = pack_products_into_restrictions(best_items, box)

        print(f"优化耗时: {datetime.datetime.now() - start}")
        print(f"空间利用率: {ratio:.2%}")

        visualize_float_packing(box, best_items, positions, i + 1)

        plt.plot(history)
        plt.title(f"Optimization Process for Box {i + 1}")
        plt.xlabel("Iteration")
        plt.ylabel("Space Ratio")
        plt.show()