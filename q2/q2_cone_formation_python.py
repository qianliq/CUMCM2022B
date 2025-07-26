"""
锥形编队问题的Python实现
基于MATLAB代码转换而来

主要功能：
1. 锥形编队位置优化
2. 基于三角定位的飞机位置调整
3. 迭代优化找到最佳调度方案
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import pandas as pd

class ConeFormationOptimizer:
    def __init__(self):
        # 标准锥形编队位置
        self.std_positions = np.array([
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
            [0, 0],   # 13 (原点)
            [0, -2],  # 14
            [0, -4]   # 15
        ])
        
        # 不需要的组合（已知不可行的三机组合）
        self.no_need_combinations = np.array([
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
        
        # 需要调度的飞机编号（0-based）
        self.need_planes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
        
        # 参数设置
        self.sigma = 50  # 标准间距
        self.new_sigma = 45  # 新间距
        self.plane_num = 14  # 飞机数量
        self.lmd = np.array([1, 1])  # 移动步长控制
        self.max_iter = 50  # 最大迭代次数
        
        # 坐标系数
        self.one_x = np.sqrt(3) / 2
        self.one_y = 1 / 2
        self.std_coords = np.array([self.one_x, self.one_y])
        
    def get_valid_combinations(self):
        """获取有效的三机组合"""
        # 生成所有可能的三机组合
        all_combinations = list(combinations(self.need_planes, 3))
        valid_combinations = []
        
        for combo in all_combinations:
            combo_array = np.array(combo)
            is_valid = True
            
            # 检查是否在不需要的组合中
            for no_need in self.no_need_combinations:
                if np.array_equal(np.sort(combo_array), np.sort(no_need)):
                    is_valid = False
                    break
            
            if is_valid:
                valid_combinations.append(combo_array)
                
        return np.array(valid_combinations)
    
    def get_random_positions(self, ratio=0.1):
        """生成带随机偏差的飞机位置"""
        noise = (np.random.rand(*self.std_positions.shape) * 2 - 1) * ratio
        return self.std_positions + noise
    
    def solve_line_params(self, M, A):
        """求解过点M和A的直线参数 ax + by + c = 0"""
        if abs(A[0] - M[0]) < 1e-10:  # 垂直线
            return np.array([1, 0])
        else:
            slope = (A[1] - M[1]) / (A[0] - M[0])
            # y - M[1] = slope * (x - M[0])
            # slope * x - y + M[1] - slope * M[0] = 0
            return np.array([slope, -1])
    
    def solve_line_intersect(self, line1, line2):
        """求解两直线交点"""
        # line1: [a1, b1, c1]
        # line2: [a2, b2, c2]
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None  # 平行线
        
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return np.array([x, y])
    
    def calc_alpha(self, A, M, O):
        """计算角度alpha"""
        vec_MA = A - M
        vec_MO = O - M
        
        # 计算夹角
        cos_alpha = np.dot(vec_MA, vec_MO) / (np.linalg.norm(vec_MA) * np.linalg.norm(vec_MO))
        cos_alpha = np.clip(cos_alpha, -1, 1)  # 防止数值误差
        alpha = np.arccos(cos_alpha)
        return alpha
    
    def cos_error(self, X, std_points, alphas):
        """计算位置误差（余弦误差）"""
        error = 0
        O = np.array([0, 0])
        
        for i, (std_point, alpha) in enumerate(zip(std_points, alphas)):
            vec_XO = O - X
            vec_X_std = std_point - X
            
            if np.linalg.norm(vec_XO) < 1e-10 or np.linalg.norm(vec_X_std) < 1e-10:
                continue
                
            cos_current = np.dot(vec_XO, vec_X_std) / (np.linalg.norm(vec_XO) * np.linalg.norm(vec_X_std))
            cos_current = np.clip(cos_current, -1, 1)
            current_alpha = np.arccos(cos_current)
            
            error += abs(current_alpha - alpha)
            
        return error
    
    def optimize_single_plane(self, plane_idx, combination, positions, std_ideal_pos):
        """优化单个飞机位置"""
        pn = self.need_planes[plane_idx]
        O = np.array([0, 0])
        
        # 获取三个参考点
        std_A = std_ideal_pos[combination[0]]
        std_B = std_ideal_pos[combination[1]]
        std_C = std_ideal_pos[combination[2]]
        
        A = positions[combination[0]]
        B = positions[combination[1]]
        C = positions[combination[2]]
        M = positions[pn]
        
        # 求解直线参数
        la_params = self.solve_line_params(M, A)
        lb_params = self.solve_line_params(M, B)
        lc_params = self.solve_line_params(M, C)
        
        # 计算直线常数项
        la_C = -la_params[0] * std_A[0] - la_params[1] * std_A[1]
        lb_C = -lb_params[0] * std_B[0] - lb_params[1] * std_B[1]
        lc_C = -lc_params[0] * std_C[0] - lc_params[1] * std_C[1]
        
        # 求交点
        line_a = np.array([la_params[0], la_params[1], la_C])
        line_b = np.array([lb_params[0], lb_params[1], lb_C])
        line_c = np.array([lc_params[0], lc_params[1], lc_C])
        
        intersect_ab = self.solve_line_intersect(line_a, line_b)
        intersect_bc = self.solve_line_intersect(line_b, line_c)
        intersect_ac = self.solve_line_intersect(line_a, line_c)
        
        # 候选点
        candidates = []
        if intersect_ab is not None:
            candidates.append(intersect_ab)
        if intersect_bc is not None:
            candidates.append(intersect_bc)
        if intersect_ac is not None:
            candidates.append(intersect_ac)
        
        if candidates:
            candidates.append(np.mean(candidates, axis=0))
        else:
            candidates = [M]  # 如果没有有效交点，使用原位置
        
        # 计算标准角度
        std_points = np.array([std_A, std_B, std_C])
        alphas = np.array([
            self.calc_alpha(A, M, O),
            self.calc_alpha(B, M, O),
            self.calc_alpha(C, M, O)
        ])
        
        # 选择最优候选点
        best_point = M
        best_error = float('inf')
        
        for candidate in candidates:
            error = self.cos_error(candidate, std_points, alphas)
            if error < best_error:
                best_error = error
                best_point = candidate
        
        return best_point, best_error
    
    def single_iteration(self, positions, std_ideal_pos, combinations):
        """单次迭代优化"""
        total_distances = np.zeros((len(combinations), self.plane_num))
        all_positions = []
        
        for comb_idx, combination in enumerate(combinations):
            new_positions = positions.copy()
            distances = np.zeros(self.plane_num)
            
            for plane_idx in range(self.plane_num):
                pn = self.need_planes[plane_idx]
                
                # 优化飞机位置
                optimal_point, _ = self.optimize_single_plane(
                    plane_idx, combination, new_positions, std_ideal_pos
                )
                
                # 计算移动向量
                target_pos = std_ideal_pos[pn]
                current_pos = new_positions[pn]
                move_vector = target_pos - optimal_point
                move_vector = self.lmd * move_vector
                
                # 如果不是调度飞机则移动
                if pn not in combination:
                    new_positions[pn] = current_pos + move_vector
                
                # 计算误差
                distance = np.linalg.norm(new_positions[pn] - target_pos)
                distances[plane_idx] = distance
            
            total_distances[comb_idx] = distances
            all_positions.append(new_positions.copy())
        
        # 选择最优组合
        sum_distances = np.sum(total_distances, axis=1)
        best_idx = np.argmin(sum_distances)
        
        return (combinations[best_idx], sum_distances[best_idx], 
                all_positions[best_idx], total_distances[best_idx])
    
    def optimize(self, noise_ratio=0.1):
        """主优化函数"""
        # 初始化位置
        np.random.seed(1)
        random_positions = self.get_random_positions(noise_ratio)
        
        # 计算实际坐标
        coords = self.sigma * self.std_coords
        std_ideal_pos = self.std_positions * coords
        positions = random_positions * coords
        
        # 获取有效组合
        combinations = self.get_valid_combinations()
        print(f"有效组合数量: {len(combinations)}")
        
        # 迭代优化
        results = {
            'combinations': [],
            'errors': [],
            'positions': [],
            'detail_errors': []
        }
        
        current_positions = positions.copy()
        
        for iteration in range(self.max_iter):
            print(f"迭代 {iteration + 1}/{self.max_iter}")
            
            best_combination, best_error, best_positions, detail_errors = \
                self.single_iteration(current_positions, std_ideal_pos, combinations)
            
            results['combinations'].append(best_combination)
            results['errors'].append(best_error)
            results['positions'].append(best_positions.copy())
            results['detail_errors'].append(detail_errors)
            
            current_positions = best_positions
            
            print(f"最优组合: {best_combination + 1}, 总误差: {best_error:.4f}")
        
        return results
    
    def plot_positions(self, positions, title="锥形编队位置"):
        """绘制编队位置"""
        plt.figure(figsize=(10, 8))
        
        # 添加原点位置
        plot_positions = positions.copy()
        if len(plot_positions) == 14:
            # 插入原点位置
            plot_positions = np.insert(plot_positions, 12, [0, 0], axis=0)
        
        plt.scatter(plot_positions[:, 0], plot_positions[:, 1], 
                   s=100, marker='+', c='black', linewidth=2)
        
        # 添加编号
        for i, pos in enumerate(plot_positions):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.show()
    
    def plot_error_convergence(self, errors):
        """绘制误差收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2, marker='o')
        plt.xlabel('迭代次数')
        plt.ylabel('总误差')
        plt.title('误差收敛曲线')
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    """主函数"""
    print("=== 锥形编队优化算法 ===")
    
    # 创建优化器
    optimizer = ConeFormationOptimizer()
    
    # 执行优化
    results = optimizer.optimize(noise_ratio=0.1)
    
    # 显示最终结果
    final_error = results['errors'][-1]
    final_combination = results['combinations'][-1]
    final_positions = results['positions'][-1]
    
    print(f"\n=== 最终结果 ===")
    print(f"最优调度组合: {final_combination + 1}")
    print(f"最终总误差: {final_error:.4f}")
    
    # 绘制结果
    optimizer.plot_positions(final_positions, "最终锥形编队位置")
    optimizer.plot_error_convergence(results['errors'])
    
    # 保存结果
    np.save('optimization_results.npy', results)
    print("结果已保存到 optimization_results.npy")

if __name__ == "__main__":
    main()
