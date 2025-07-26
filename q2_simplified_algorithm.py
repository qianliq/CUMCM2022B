"""
锥形编队优化算法 - 核心思路简化版
更清晰地展示算法的核心逻辑
"""

import numpy as np
import matplotlib.pyplot as plt
from q4_helper_functions import *

class SimplifiedConeOptimizer:
    def __init__(self):
        """初始化优化器"""
        # 基本参数
        self.sigma = 50  # 标准间距
        self.plane_num = 14  # 需要调整的飞机数量
        self.lmd = np.array([1.0, 1.0])  # 移动步长控制
        
        # 坐标变换参数
        self.coords_factor = np.array([np.sqrt(3)/2, 0.5])
        
        # 需要调整的飞机索引（0-based）
        self.need_planes = np.array([0,1,2,3,4,5,6,7,8,9,10,11,13,14])
        
    def single_plane_optimization(self, plane_idx, reference_combo, positions, std_ideal_pos):
        """
        单个飞机位置优化的核心算法
        
        参数:
        plane_idx: 要优化的飞机在need_planes中的索引
        reference_combo: 参考的三架飞机索引 [i, j, k]
        positions: 当前所有飞机位置
        std_ideal_pos: 理想位置
        
        返回:
        optimal_position: 优化后的位置
        error: 位置误差
        """
        # 获取飞机编号
        plane_id = self.need_planes[plane_idx]
        
        # 原点
        O = np.array([0, 0])
        
        # ===== 步骤1: 获取参考点信息 =====
        # 三个参考点的理想位置
        std_A = std_ideal_pos[reference_combo[0]]
        std_B = std_ideal_pos[reference_combo[1]] 
        std_C = std_ideal_pos[reference_combo[2]]
        
        # 三个参考点的实际位置
        A = positions[reference_combo[0]]
        B = positions[reference_combo[1]]
        C = positions[reference_combo[2]]
        
        # 当前飞机位置
        M = positions[plane_id]
        
        print(f"  优化飞机{plane_id+1}, 参考飞机: {reference_combo+1}")
        
        # ===== 步骤2: 计算角度约束 =====
        # 计算M到各参考点的角度
        alpha_A = calc_alpha(A, M, O)
        alpha_B = calc_alpha(B, M, O) 
        alpha_C = calc_alpha(C, M, O)
        
        # ===== 步骤3: 构建几何约束方程 =====
        # 过M和各参考点的直线参数
        la_params = solve_AB(M, A)
        lb_params = solve_AB(M, B)
        lc_params = solve_AB(M, C)
        
        # 计算直线常数项（使直线过对应的理想点）
        la_C = -la_params[0] * std_A[0] - la_params[1] * std_A[1]
        lb_C = -lb_params[0] * std_B[0] - lb_params[1] * std_B[1]
        lc_C = -lc_params[0] * std_C[0] - lc_params[1] * std_C[1]
        
        # ===== 步骤4: 求解候选位置 =====
        # 构建直线方程
        line_a = np.array([la_params[0], la_params[1], la_C])
        line_b = np.array([lb_params[0], lb_params[1], lb_C])
        line_c = np.array([lc_params[0], lc_params[1], lc_C])
        
        # 计算三条直线的交点
        candidates = []
        intersect_ab = solve_line_intersect(line_a, line_b)
        intersect_bc = solve_line_intersect(line_b, line_c)
        intersect_ac = solve_line_intersect(line_a, line_c)
        
        # 收集有效的候选点
        if intersect_ab is not None:
            candidates.append(intersect_ab)
        if intersect_bc is not None:
            candidates.append(intersect_bc)
        if intersect_ac is not None:
            candidates.append(intersect_ac)
        
        # 添加候选点的重心作为额外选项
        if candidates:
            centroid = np.mean(candidates, axis=0)
            candidates.append(centroid)
        else:
            candidates = [M]  # 如果没有有效交点，保持原位置
        
        # ===== 步骤5: 选择最优候选点 =====
        std_points = np.array([std_A, std_B, std_C])
        target_alphas = np.array([alpha_A, alpha_B, alpha_C])
        
        best_position = M
        best_error = float('inf')
        
        for candidate in candidates:
            error = cos_error(candidate, std_points, target_alphas)
            if error < best_error:
                best_error = error
                best_position = candidate
        
        print(f"    最优候选点: {best_position}, 误差: {best_error:.6f}")
        
        return best_position, best_error
    
    def single_iteration(self, positions, std_ideal_pos):
        """
        单次迭代优化
        
        参数:
        positions: 当前飞机位置
        std_ideal_pos: 理想飞机位置
        
        返回:
        best_combo: 最优参考组合
        best_error: 最小总误差
        best_positions: 最优位置配置
        """
        print("\n--- 开始新的迭代 ---")
        
        # 获取所有有效的三机组合
        valid_combinations = delete_no_need_combinations()
        print(f"有效组合数: {len(valid_combinations)}")
        
        best_combo = None
        best_error = float('inf')
        best_positions = positions.copy()
        
        # 遍历所有有效组合
        for combo_idx, combination in enumerate(valid_combinations):
            if combo_idx % 50 == 0:
                print(f"正在评估组合 {combo_idx+1}/{len(valid_combinations)}")
            
            current_positions = positions.copy()
            total_distance = 0.0
            
            # 对每架需要调整的飞机进行优化
            for plane_idx in range(self.plane_num):
                plane_id = self.need_planes[plane_idx]
                
                # 计算最优位置
                optimal_pos, _ = self.single_plane_optimization(
                    plane_idx, combination, current_positions, std_ideal_pos
                )
                
                # ===== 位置更新策略 =====
                target_pos = std_ideal_pos[plane_id]
                current_pos = current_positions[plane_id]
                
                # 计算移动向量
                movement_vector = target_pos - optimal_pos
                controlled_movement = self.lmd * movement_vector
                
                # 只有非参考飞机才移动
                if plane_id not in combination:
                    new_position = current_pos + controlled_movement
                    current_positions[plane_id] = new_position
                
                # 计算到目标位置的距离误差
                final_error = np.linalg.norm(current_positions[plane_id] - target_pos)
                total_distance += final_error
            
            # 更新最优解
            if total_distance < best_error:
                best_error = total_distance
                best_combo = combination
                best_positions = current_positions.copy()
        
        print(f"最优组合: {best_combo + 1}, 总误差: {best_error:.6f}")
        return best_combo, best_error, best_positions
    
    def optimize(self, max_iterations=10, noise_ratio=0.1):
        """
        主优化函数
        
        参数:
        max_iterations: 最大迭代次数
        noise_ratio: 初始位置噪声比例
        
        返回:
        optimization_results: 优化结果字典
        """
        print("=== 锥形编队优化算法（简化版）===")
        
        # ===== 初始化 =====
        # 设置随机种子确保可重复性
        np.random.seed(42)
        
        # 生成初始位置（带噪声）
        std_positions = np.array([get_conical_plane_pos(i) for i in range(1, 16)])
        noisy_positions = np.array([get_rand_con_plane_pos(i, noise_ratio) for i in range(1, 16)])
        
        # 坐标变换
        coords = self.sigma * self.coords_factor
        std_ideal_pos = std_positions * coords  
        current_positions = noisy_positions * coords
        
        print(f"初始噪声比例: {noise_ratio*100}%")
        print(f"坐标变换系数: {coords}")
        
        # ===== 迭代优化 =====
        results = {
            'combinations': [],
            'errors': [],
            'positions': [],
            'iteration_details': []
        }
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
            
            # 执行单次迭代
            best_combo, best_error, best_positions = self.single_iteration(
                current_positions, std_ideal_pos
            )
            
            # 记录结果
            results['combinations'].append(best_combo)
            results['errors'].append(best_error)
            results['positions'].append(best_positions.copy())
            
            # 更新当前位置
            current_positions = best_positions
            
            print(f"\n迭代{iteration + 1}完成:")
            print(f"  最优组合: 飞机{best_combo + 1}")
            print(f"  总误差: {best_error:.6f}")
            
            # 检查收敛性
            if best_error < 1e-6:
                print(f"\n算法在第{iteration + 1}轮迭代后收敛！")
                break
        
        # ===== 输出最终结果 =====
        print(f"\n{'='*50}")
        print("优化完成！")
        print(f"{'='*50}")
        print(f"最终最优组合: 飞机{results['combinations'][-1] + 1}")
        print(f"最终总误差: {results['errors'][-1]:.8f}")
        
        return results
    
    def visualize_results(self, results):
        """可视化优化结果"""
        # 绘制误差收敛曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(results['errors']) + 1), results['errors'], 
                'b-o', linewidth=2, markersize=6)
        plt.xlabel('迭代次数')
        plt.ylabel('总误差')
        plt.title('误差收敛曲线')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 绘制最终位置
        plt.subplot(1, 2, 2)
        final_positions = results['positions'][-1]
        
        # 添加原点
        plot_positions = final_positions.copy()
        plot_positions = np.insert(plot_positions, 12, [0, 0], axis=0)
        
        plt.scatter(plot_positions[:, 0], plot_positions[:, 1], 
                   s=100, marker='+', c='red', linewidth=2)
        
        for i, pos in enumerate(plot_positions):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('最终编队位置')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数演示"""
    # 创建优化器
    optimizer = SimplifiedConeOptimizer()
    
    # 运行优化（少量迭代用于演示）
    results = optimizer.optimize(max_iterations=5, noise_ratio=0.05)
    
    # 可视化结果
    optimizer.visualize_results(results)

if __name__ == "__main__":
    main()
