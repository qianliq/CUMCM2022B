"""
锥形编队优化算法的辅助函数
包含MATLAB代码中的关键函数的Python实现
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def solve_AB(M, A):
    """
    求解过点M和A的直线参数
    返回直线方程 ax + by = 0 的系数 [a, b]
    对应MATLAB中的solveAB函数
    """
    if abs(A[0] - M[0]) < 1e-10:  # 垂直线
        return np.array([1, 0])
    else:
        # 斜率
        k = (A[1] - M[1]) / (A[0] - M[0])
        # 直线方程: y - M[1] = k(x - M[0])
        # 即: kx - y + M[1] - k*M[0] = 0
        # 标准化为: ax + by = 0 形式
        return np.array([k, -1])

def solve_line_intersect(line1, line2):
    """
    求解两条直线的交点
    line1: [a1, b1, c1] 表示 a1*x + b1*y + c1 = 0
    line2: [a2, b2, c2] 表示 a2*x + b2*y + c2 = 0
    对应MATLAB中的solveLineInsect函数
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    # 计算行列式
    det = a1 * b2 - a2 * b1
    
    if abs(det) < 1e-10:
        return None  # 平行线，没有交点
    
    # 使用克拉默法则求解
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    
    return np.array([x, y])

def calc_alpha(A, M, O):
    """
    计算角度alpha = ∠AMO
    对应MATLAB中的calAlpha函数
    """
    # 向量MA和MO
    vec_MA = A - M
    vec_MO = O - M
    
    # 计算向量的模长
    norm_MA = np.linalg.norm(vec_MA)
    norm_MO = np.linalg.norm(vec_MO)
    
    if norm_MA < 1e-10 or norm_MO < 1e-10:
        return 0.0
    
    # 计算夹角的余弦值
    cos_alpha = np.dot(vec_MA, vec_MO) / (norm_MA * norm_MO)
    
    # 防止数值误差导致的定义域错误
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    
    # 返回角度（弧度）
    alpha = np.arccos(cos_alpha)
    return alpha

def cos_error(X, std_points, alphas):
    """
    计算位置X的角度误差
    对应MATLAB中的cosError函数
    
    参数:
    X: 当前位置 [x, y]
    std_points: 标准参考点位置 [[x1,y1], [x2,y2], [x3,y3]]
    alphas: 目标角度 [alpha1, alpha2, alpha3]
    """
    total_error = 0.0
    O = np.array([0, 0])  # 原点
    
    for i, (std_point, target_alpha) in enumerate(zip(std_points, alphas)):
        # 计算当前角度
        current_alpha = calc_alpha(std_point, X, O)
        
        # 累加角度误差
        error = abs(current_alpha - target_alpha)
        total_error += error
    
    return total_error

def unlinear_constraint(X, params, Ces):
    """
    非线性约束函数
    对应MATLAB中的unlinearC函数
    
    参数:
    X: 位置变量 [x, y]
    params: 直线参数矩阵 3x2
    Ces: 直线常数项 [c1, c2, c3]
    """
    # 计算约束条件
    constraints = X.dot(params.T) + Ces  # 1x3
    
    # 不等式约束: 所有约束的乘积 <= 0
    c = np.prod(constraints)
    
    # 等式约束（无）
    ceq = []
    
    return c, ceq

def get_conical_plane_pos(number):
    """
    获取锥形编队中第number号飞机的标准位置
    对应MATLAB中的getConicalPlanesPos函数
    """
    X = np.array([
        [4, 0],   # 1
        [3, 1],   # 2
        [3, -1],  # 3
        [2, 2],   # 4
        [2, 0],   # 5
        [2, -2],  # 6
        [1, 3],   # 7
        [1, 1],   # 8
        [1, -1],  # 9
        [1, -3],  # 10
        [0, 4],   # 11
        [0, 2],   # 12
        [0, 0],   # 13
        [0, -2],  # 14
        [0, -4]   # 15
    ])
    
    return X[number - 1]  # 转换为0-based索引

def get_rand_con_plane_pos(number, ratio=0.01):
    """
    获取带随机误差的飞机位置
    对应MATLAB中的getRandConPlanePos函数
    
    参数:
    number: 飞机编号 (1-15)
    ratio: 误差比例 (默认1%)
    """
    # 获取标准位置
    std_pos = get_conical_plane_pos(number)
    
    # 添加随机误差
    noise = (np.random.rand(2) * 2 - 1) * ratio  # [-ratio, ratio]
    
    return std_pos + noise

def plot_positions_conical(positions):
    """
    绘制锥形编队位置
    对应MATLAB中的plotPositionsConinal函数
    """
    plt.figure(figsize=(10, 8))
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    # 绘制散点图
    plt.scatter(x, y, s=100, marker='+', c='black', linewidth=2)
    
    # 添加编号标签
    for i, (xi, yi) in enumerate(positions):
        plt.annotate(f'{i+1}', (xi, yi), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    # 设置图形属性
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title('Conical Formation Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def delete_no_need_combinations():
    """
    生成有效的三机组合，排除不需要的组合
    对应MATLAB中的deleteNoNeedFromCms.m文件功能
    """
    # 不需要的组合（1-based索引）
    no_need = np.array([
        [1, 2, 3],
        [1, 4, 6],
        [1, 7, 10],
        [1, 11, 15],
        [1, 12, 14],
        [1, 8, 9],
        [5, 2, 3],
        [5, 4, 6],
        [5, 7, 10],
        [5, 11, 15],
        [5, 12, 14],
        [5, 8, 9]
    ]) - 1  # 转换为0-based索引
    
    # 需要的飞机编号（0-based）
    need_plans_plane = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
    
    # 生成所有可能的三机组合
    from itertools import combinations
    all_combinations = list(combinations(need_plans_plane, 3))
    
    # 筛选有效组合
    valid_combinations = []
    
    for combo in all_combinations:
        combo_array = np.array(combo)
        is_valid = True
        
        # 检查是否在不需要的组合中
        for no_need_combo in no_need:
            if np.array_equal(np.sort(combo_array), np.sort(no_need_combo)):
                is_valid = False
                break
        
        if is_valid:
            valid_combinations.append(combo_array)
    
    return np.array(valid_combinations)

# 示例使用
def demo_functions():
    """演示各个函数的使用"""
    print("=== 锥形编队优化算法辅助函数演示 ===\n")
    
    # 1. 获取标准位置
    print("1. 标准位置:")
    for i in range(1, 6):
        pos = get_conical_plane_pos(i)
        print(f"飞机{i}: {pos}")
    
    # 2. 获取带误差的位置
    print("\n2. 带误差位置:")
    np.random.seed(42)
    for i in range(1, 6):
        pos = get_rand_con_plane_pos(i, ratio=0.05)
        print(f"飞机{i}: {pos}")
    
    # 3. 角度计算
    print("\n3. 角度计算:")
    M = np.array([1, 1])
    A = np.array([2, 2])
    O = np.array([0, 0])
    alpha = calc_alpha(A, M, O)
    print(f"角度∠AMO = {np.degrees(alpha):.2f}°")
    
    # 4. 直线参数
    print("\n4. 直线参数:")
    line_params = solve_AB(M, A)
    print(f"过点M{M}和A{A}的直线参数: {line_params}")
    
    # 5. 有效组合数量
    valid_combos = delete_no_need_combinations()
    print(f"\n5. 有效三机组合数量: {len(valid_combos)}")
    
    # 6. 绘制标准编队
    print("\n6. 绘制标准编队位置")
    std_positions = np.array([get_conical_plane_pos(i) for i in range(1, 16)])
    plot_positions_conical(std_positions)

if __name__ == "__main__":
    demo_functions()
