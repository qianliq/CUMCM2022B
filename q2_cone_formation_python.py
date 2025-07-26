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
        """Main optimization function"""
        # Initialize positions
        np.random.seed(1)
        random_positions = self.get_random_positions(noise_ratio)
        
        # Calculate actual coordinates
        coords = self.sigma * self.std_coords
        std_ideal_pos = self.std_positions * coords
        positions = random_positions * coords
        
        # Store initial positions for comparison
        initial_positions = positions.copy()
        
        # Get valid combinations
        combinations = self.get_valid_combinations()
        print(f"Number of valid combinations: {len(combinations)}")
        
        # Iterative optimization
        results = {
            'combinations': [],
            'errors': [],
            'positions': [],
            'detail_errors': [],
            'initial_positions': initial_positions
        }
        
        current_positions = positions.copy()
        
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}/{self.max_iter}")
            
            best_combination, best_error, best_positions, detail_errors = \
                self.single_iteration(current_positions, std_ideal_pos, combinations)
            
            results['combinations'].append(best_combination)
            results['errors'].append(best_error)
            results['positions'].append(best_positions.copy())
            results['detail_errors'].append(detail_errors)
            
            current_positions = best_positions
            
            print(f"Best combination: {best_combination + 1}, Total error: {best_error:.4f}")
        
        return results
    
    def plot_positions(self, positions, title="Cone Formation Position"):
        """Plot formation positions"""
        plt.figure(figsize=(10, 8))
        
        # Add origin position
        plot_positions = positions.copy()
        if len(plot_positions) == 14:
            # Insert origin position
            plot_positions = np.insert(plot_positions, 12, [0, 0], axis=0)
        
        plt.scatter(plot_positions[:, 0], plot_positions[:, 1], 
                   s=100, marker='+', c='black', linewidth=2)
        
        # Add numbering
        for i, pos in enumerate(plot_positions):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    
    def plot_error_convergence(self, errors):
        """Plot error convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2, marker='o')
        plt.xlabel('Iteration Number')
        plt.ylabel('Total Error')
        plt.title('Error Convergence Curve')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_comparison(self, initial_positions, final_positions, title="Formation Comparison"):
        """Plot comparison between initial, final and standard positions"""
        plt.figure(figsize=(15, 5))
        
        # Calculate standard positions
        coords = self.sigma * self.std_coords
        std_ideal_pos = self.std_positions * coords
        
        # Add origin for plotting (position 13)
        def add_origin(positions):
            if len(positions) == 14:
                return np.insert(positions, 12, [0, 0], axis=0)
            return positions
        
        plot_initial = add_origin(initial_positions)
        plot_final = add_origin(final_positions)
        plot_standard = add_origin(std_ideal_pos)
        
        # Subplot 1: Initial positions
        plt.subplot(1, 3, 1)
        plt.scatter(plot_initial[:, 0], plot_initial[:, 1], 
                   s=100, marker='o', c='red', alpha=0.7, label='Initial Position')
        for i, pos in enumerate(plot_initial):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Initial Positions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        
        # Subplot 2: Final positions
        plt.subplot(1, 3, 2)
        plt.scatter(plot_final[:, 0], plot_final[:, 1], 
                   s=100, marker='s', c='blue', alpha=0.7, label='Final Position')
        for i, pos in enumerate(plot_final):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Final Positions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        
        # Subplot 3: Standard cone formation
        plt.subplot(1, 3, 3)
        plt.scatter(plot_standard[:, 0], plot_standard[:, 1], 
                   s=100, marker='^', c='green', alpha=0.7, label='Standard Cone')
        for i, pos in enumerate(plot_standard):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Standard Cone Formation')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.show()
    
    def draw_cone_connections(self, positions, color='green', alpha=0.3, linewidth=1):
        """Draw lines to show cone formation structure"""
        # Define cone formation connections based on the standard positions
        # These connections will form the cone shape
        connections = [
            # Outer boundary of the cone
            [0, 1, 3, 6, 10, 14, 9, 5, 2, 0],  # Outer perimeter
            # Inner structure layers
            [1, 4, 7, 11],  # Inner connections - left side
            [2, 4, 9, 14],  # Inner connections - right side
            [3, 7, 12],     # Cross connections
            [5, 8, 12],     # Cross connections
            [6, 9, 13],     # Cross connections
            # Central radial lines
            [0, 12],  # Top to center
            [4, 12],  # Middle to center
            [7, 8],   # Inner horizontal connection
            [8, 9],   # Inner horizontal connection
        ]
        
        for connection in connections:
            for i in range(len(connection) - 1):
                start_idx = connection[i]
                end_idx = connection[i + 1]
                plt.plot([positions[start_idx, 0], positions[end_idx, 0]], 
                        [positions[start_idx, 1], positions[end_idx, 1]], 
                        color=color, alpha=alpha, linewidth=linewidth, linestyle='-')
    
    def plot_individual_charts(self, initial_positions, final_positions, errors, 
                             save_dir="output_figures"):
        """Plot individual charts one by one and save them"""
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # Calculate standard positions
        coords = self.sigma * self.std_coords
        std_ideal_pos = self.std_positions * coords
        
        # Add origin for plotting (position 13)
        def add_origin(positions):
            if len(positions) == 14:
                return np.insert(positions, 12, [0, 0], axis=0)
            return positions
        
        plot_initial = add_origin(initial_positions)
        plot_final = add_origin(final_positions)
        plot_standard = add_origin(std_ideal_pos)
        
        # Chart 1: Initial positions
        print("Displaying Chart 1: Initial Positions...")
        plt.figure(figsize=(10, 8))
        plt.scatter(plot_initial[:, 0], plot_initial[:, 1], 
                   s=120, marker='o', c='red', alpha=0.7, label='Initial Position',
                   edgecolor='darkred', linewidth=1.5)
        for i, pos in enumerate(plot_initial):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        plt.xlim([-130, 130])
        plt.ylim([-130, 130])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Initial Aircraft Positions', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/01_initial_positions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 2: Final positions
        print("Displaying Chart 2: Final Positions...")
        plt.figure(figsize=(10, 8))
        plt.scatter(plot_final[:, 0], plot_final[:, 1], 
                   s=120, marker='s', c='blue', alpha=0.7, label='Final Position',
                   edgecolor='darkblue', linewidth=1.5)
        for i, pos in enumerate(plot_final):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        plt.xlim([-130, 130])
        plt.ylim([-130, 130])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Final Aircraft Positions', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/02_final_positions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 3: Standard cone formation with connections
        print("Displaying Chart 3: Standard Cone Formation...")
        plt.figure(figsize=(10, 8))
        # Draw cone connections first
        self.draw_cone_connections(plot_standard, color='darkgreen', alpha=0.4, linewidth=2)
        plt.scatter(plot_standard[:, 0], plot_standard[:, 1], 
                   s=140, marker='^', c='green', alpha=0.8, label='Standard Cone', 
                   edgecolor='darkgreen', linewidth=2)
        for i, pos in enumerate(plot_standard):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, 
                        fontweight='bold', color='darkgreen')
        plt.xlim([-130, 130])
        plt.ylim([-130, 130])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Standard Cone Formation', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/03_standard_cone_formation.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 4: Movement comparison with arrows
        print("Displaying Chart 4: Movement Trajectory...")
        plt.figure(figsize=(12, 10))
        # Draw standard cone connections as background
        self.draw_cone_connections(plot_standard, color='lightgreen', alpha=0.2, linewidth=1)
        
        # Plot all three sets of positions
        plt.scatter(plot_initial[:, 0], plot_initial[:, 1], 
                   s=120, marker='o', c='red', alpha=0.6, label='Initial Position', 
                   edgecolor='darkred', linewidth=1.5)
        plt.scatter(plot_final[:, 0], plot_final[:, 1], 
                   s=100, marker='s', c='blue', alpha=0.7, label='Final Position', 
                   edgecolor='darkblue', linewidth=1.5)
        plt.scatter(plot_standard[:, 0], plot_standard[:, 1], 
                   s=80, marker='^', c='green', alpha=0.8, label='Standard Position', 
                   edgecolor='darkgreen', linewidth=1.5)
        
        # Draw arrows from initial to final positions
        for i in range(len(plot_initial)):
            plt.arrow(plot_initial[i, 0], plot_initial[i, 1],
                     plot_final[i, 0] - plot_initial[i, 0],
                     plot_final[i, 1] - plot_initial[i, 1],
                     head_width=3, head_length=3, fc='gray', ec='gray', alpha=0.6)
        
        plt.xlim([-130, 130])
        plt.ylim([-130, 130])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Aircraft Movement Trajectory', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/04_movement_trajectory.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 5: Error convergence
        print("Displaying Chart 5: Error Convergence...")
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=3, marker='o', 
                markersize=6, markerfacecolor='lightblue', markeredgecolor='darkblue')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Total Error', fontsize=12)
        plt.title('Optimization Error Convergence Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key points
        min_error_idx = np.argmin(errors)
        plt.annotate(f'Min Error: {min(errors):.4f}\nIteration: {min_error_idx+1}', 
                    xy=(min_error_idx+1, min(errors)), 
                    xytext=(min_error_idx+1+10, min(errors)+max(errors)*0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/05_error_convergence.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chart 6: Final vs Standard alignment
        print("Displaying Chart 6: Final vs Standard Alignment...")
        plt.figure(figsize=(12, 10))
        # Draw standard cone connections
        self.draw_cone_connections(plot_standard, color='darkgreen', alpha=0.4, linewidth=2)
        
        plt.scatter(plot_final[:, 0], plot_final[:, 1], 
                   s=140, marker='s', c='blue', alpha=0.7, label='Final Position', 
                   edgecolor='darkblue', linewidth=2)
        plt.scatter(plot_standard[:, 0], plot_standard[:, 1], 
                   s=120, marker='^', c='green', alpha=0.8, label='Standard Cone', 
                   edgecolor='darkgreen', linewidth=2)
        
        # Draw connection lines between final and standard positions
        for i in range(len(plot_final)):
            error_distance = np.linalg.norm(plot_final[i] - plot_standard[i])
            # Color code the error lines
            if error_distance < 2.0:
                line_color = 'green'
                alpha_val = 0.3
            elif error_distance < 5.0:
                line_color = 'orange'
                alpha_val = 0.5
            else:
                line_color = 'red'
                alpha_val = 0.7
                
            plt.plot([plot_final[i, 0], plot_standard[i, 0]], 
                    [plot_final[i, 1], plot_standard[i, 1]], 
                    color=line_color, alpha=alpha_val, linewidth=2, linestyle='--')
        
        # Add numbering
        for i, pos in enumerate(plot_standard):
            plt.annotate(f'{i+1}', (pos[0], pos[1]), 
                        xytext=(8, 8), textcoords='offset points', fontsize=10, 
                        fontweight='bold', color='darkgreen')
        
        plt.xlim([-130, 130])
        plt.ylim([-130, 130])
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Final vs Standard Position Alignment', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.legend(fontsize=12)
        
        # Add color legend for error lines
        plt.text(0.02, 0.98, 'Error Lines:\nGreen: < 2.0\nOrange: 2.0-5.0\nRed: > 5.0', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/06_final_vs_standard_alignment.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nAll charts have been saved to the '{save_dir}' directory:")
        print("  01_initial_positions.png")
        print("  02_final_positions.png") 
        print("  03_standard_cone_formation.png")
        print("  04_movement_trajectory.png")
        print("  05_error_convergence.png")
        print("  06_final_vs_standard_alignment.png")
    
    def analyze_results(self, initial_positions, final_positions, final_error, final_combination):
        """Analyze and display detailed results"""
        # Calculate standard positions
        coords = self.sigma * self.std_coords
        std_ideal_pos = self.std_positions * coords
        
        # Add origin for comparison (position 13)
        def add_origin(positions):
            if len(positions) == 14:
                return np.insert(positions, 12, [0, 0], axis=0)
            return positions
        
        plot_initial = add_origin(initial_positions)
        plot_final = add_origin(final_positions)
        plot_standard = add_origin(std_ideal_pos)
        
        print("\n" + "="*80)
        print("DETAILED FORMATION ANALYSIS")
        print("="*80)
        
        print(f"\nOptimal scheduling combination: Aircraft {final_combination + 1}")
        print(f"Final total error: {final_error:.4f}")
        
        print("\n" + "-"*80)
        print("INDIVIDUAL AIRCRAFT ANALYSIS")
        print("-"*80)
        print(f"{'Aircraft':<8} {'Initial Pos':<15} {'Final Pos':<15} {'Standard Pos':<15} {'Movement':<12} {'Error':<10} {'Match %':<8}")
        print("-"*80)
        
        total_movement = 0
        total_error = 0
        perfect_matches = 0
        
        for i in range(15):  # 15 aircraft total
            initial_pos = plot_initial[i]
            final_pos = plot_final[i]
            standard_pos = plot_standard[i]
            
            # Calculate movement distance
            movement = np.linalg.norm(final_pos - initial_pos)
            total_movement += movement
            
            # Calculate error (distance from standard)
            error = np.linalg.norm(final_pos - standard_pos)
            total_error += error
            
            # Calculate match percentage (closer to 100% means better match)
            max_possible_error = np.linalg.norm(standard_pos) + 50  # Rough estimate
            match_percentage = max(0, (1 - error / max_possible_error) * 100)
            
            if error < 1.0:  # Consider very close as perfect match
                perfect_matches += 1
            
            print(f"{i+1:<8} "
                  f"({initial_pos[0]:6.1f},{initial_pos[1]:6.1f}) "
                  f"({final_pos[0]:6.1f},{final_pos[1]:6.1f}) "
                  f"({standard_pos[0]:6.1f},{standard_pos[1]:6.1f}) "
                  f"{movement:8.2f}    "
                  f"{error:6.2f}    "
                  f"{match_percentage:5.1f}%")
        
        print("-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        print(f"Total movement distance: {total_movement:.2f}")
        print(f"Average movement per aircraft: {total_movement/15:.2f}")
        print(f"Total positioning error: {total_error:.2f}")
        print(f"Average error per aircraft: {total_error/15:.2f}")
        print(f"Aircraft with perfect matches (error < 1.0): {perfect_matches}/15")
        print(f"Overall formation accuracy: {(1 - total_error/(15*100))*100:.1f}%")
        
        print("\n" + "-"*80)
        print("FORMATION QUALITY ASSESSMENT")
        print("-"*80)
        
        # Analyze different aspects
        if total_error/15 < 2.0:
            quality = "Excellent"
        elif total_error/15 < 5.0:
            quality = "Good"
        elif total_error/15 < 10.0:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"Formation Quality: {quality}")
        
        # Check convergence efficiency
        if total_movement/15 < 10.0:
            efficiency = "Highly Efficient"
        elif total_movement/15 < 20.0:
            efficiency = "Efficient"
        elif total_movement/15 < 30.0:
            efficiency = "Moderate"
        else:
            efficiency = "Low Efficiency"
        
        print(f"Movement Efficiency: {efficiency}")
        
        # Identify best and worst performers
        errors = [np.linalg.norm(plot_final[i] - plot_standard[i]) for i in range(15)]
        best_aircraft = np.argmin(errors) + 1
        worst_aircraft = np.argmax(errors) + 1
        
        print(f"Best positioned aircraft: #{best_aircraft} (error: {min(errors):.2f})")
        print(f"Worst positioned aircraft: #{worst_aircraft} (error: {max(errors):.2f})")
        
        # Analyze the scheduled aircraft (those in final_combination)
        print(f"\nScheduled aircraft analysis (Aircraft {final_combination + 1}):")
        for aircraft_idx in final_combination:
            aircraft_num = aircraft_idx + 1
            error = errors[aircraft_idx]
            print(f"  Aircraft #{aircraft_num}: Fixed position, error = {error:.2f}")
        
        print("\n" + "="*80)

def main():
    """Main function"""
    print("=== Cone Formation Optimization Algorithm ===")
    
    # Create optimizer
    optimizer = ConeFormationOptimizer()
    
    # Execute optimization
    results = optimizer.optimize(noise_ratio=0.1)
    
    # Display final results
    final_error = results['errors'][-1]
    final_combination = results['combinations'][-1]
    final_positions = results['positions'][-1]
    initial_positions = results['initial_positions']
    
    print(f"\n=== Final Results ===")
    print(f"Optimal scheduling combination: {final_combination + 1}")
    print(f"Final total error: {final_error:.4f}")
    
    # Plot individual charts and save them
    print("\nGenerating individual visualizations...")
    optimizer.plot_individual_charts(initial_positions, final_positions, 
                                   results['errors'])
    
    # Detailed analysis
    optimizer.analyze_results(initial_positions, final_positions, final_error, final_combination)
    
    # Save results
    np.save('optimization_results.npy', results)
    print("Results saved to optimization_results.npy")

if __name__ == "__main__":
    main()
