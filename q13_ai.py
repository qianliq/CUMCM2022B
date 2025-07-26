import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import copy

# --- 1. 辅助函数 ---

def polar_to_cartesian(r, theta_deg):
    """极坐标转直角坐标"""
    theta_rad = np.radians(theta_deg)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return np.array([x, y])

def cartesian_to_polar(x, y):
    """直角坐标转极坐标"""
    r = np.sqrt(x**2 + y**2)
    theta_rad = np.arctan2(y, x)
    theta_deg = np.degrees(theta_rad)
    # 角度调整到 [0, 360)
    if theta_deg < 0:
        theta_deg += 360
    return r, theta_deg

def calculate_angle_deg(p1, p2, p3):
    """
    计算角度 p1-p2-p3 (p2是顶点)
    p1, p2, p3 是2D点 (numpy arrays)
    返回角度，单位为度
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 < 1e-12 or norm_v2 < 1e-12:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

# --- 2. 定位算法 ---

def locate_drone_trilateration(receiver_pos, transmitter_positions, measured_angles):
    """
    改进的基于角度信息的定位算法
    使用几何约束和迭代优化来调整无人机位置
    
    Args:
        receiver_pos: 接收机当前位置 [x, y]
        transmitter_positions: 发射机位置列表 [[x0,y0], [x1,y1], [x2,y2], ...]
        measured_angles: 测量到的角度列表 [angle1, angle2, ...]
    
    Returns:
        计算得到的新位置 [x, y]
    """
    if len(transmitter_positions) < 2 or len(measured_angles) < 1:
        return receiver_pos
    
    # 获取圆心位置（FY00）
    center = np.array(transmitter_positions[0])
    
    # 计算当前到圆心的距离和角度
    current_r, current_theta = cartesian_to_polar(
        receiver_pos[0] - center[0], 
        receiver_pos[1] - center[1]
    )
    
    # 如果只有一个角度信息，使用简化定位
    if len(transmitter_positions) >= 2 and len(measured_angles) >= 1:
        # 使用第一个发射机进行定位
        t1 = np.array(transmitter_positions[1])
        alpha1 = np.radians(measured_angles[0])
        
        # 计算发射机到圆心的距离和角度
        t1_r = np.linalg.norm(t1 - center)
        t1_theta_rad = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        
        # 使用正弦定理估算到圆心的距离
        # 在三角形 center-receiver-t1 中
        if np.sin(alpha1) > 1e-6:  # 避免除零
            # 使用角度约束估算新的半径
            # 这里采用一个简化的几何关系
            estimated_r = t1_r * np.sin(alpha1) / np.sin(np.pi - alpha1)
            
            # 限制半径在合理范围内
            if estimated_r > 0 and estimated_r < 200:
                # 保持当前角度，调整半径
                new_x = center[0] + estimated_r * np.cos(np.radians(current_theta))
                new_y = center[1] + estimated_r * np.sin(np.radians(current_theta))
                return np.array([new_x, new_y])
    
    # 如果有多个角度信息，使用更复杂的约束
    if len(transmitter_positions) >= 3 and len(measured_angles) >= 2:
        t1 = np.array(transmitter_positions[1])
        t2 = np.array(transmitter_positions[2])
        alpha1 = np.radians(measured_angles[0])
        alpha2 = np.radians(measured_angles[1])
        
        # 计算两个发射机的位置信息
        t1_r = np.linalg.norm(t1 - center)
        t2_r = np.linalg.norm(t2 - center)
        
        # 使用两个角度约束的加权平均
        if np.sin(alpha1) > 1e-6 and np.sin(alpha2) > 1e-6:
            r1_est = t1_r * np.sin(alpha1) / np.sin(np.pi - alpha1)
            r2_est = t2_r * np.sin(alpha2) / np.sin(np.pi - alpha2)
            
            # 取加权平均
            w1 = 1.0 / (alpha1 + 1e-6)  # 角度越小，权重越大
            w2 = 1.0 / (alpha2 + 1e-6)
            estimated_r = (w1 * r1_est + w2 * r2_est) / (w1 + w2)
            
            if estimated_r > 0 and estimated_r < 200:
                new_x = center[0] + estimated_r * np.cos(np.radians(current_theta))
                new_y = center[1] + estimated_r * np.sin(np.radians(current_theta))
                return np.array([new_x, new_y])
    
    return receiver_pos

def adjust_position_gradually(current_pos, target_pos, step_factor=0.5):
    """
    渐进式位置调整，避免过大的移动
    """
    move_vector = np.array(target_pos) - np.array(current_pos)
    adjusted_pos = np.array(current_pos) + step_factor * move_vector
    return adjusted_pos

# --- 3. 主迭代调整程序 ---

def main():
    """
    主程序：迭代+贪心算法实现无人机编队调整
    
    算法思路：
    1. 每次迭代选择最优的发射机组合（FY00 + 最多3架周边无人机）
    2. 贪心策略：选择能使总误差最小的发射机组合
    3. 对非发射机进行定位和位置调整
    4. 重复迭代直到收敛
    """
    
    # --- 初始化数据 ---
    # 表1：无人机初始位置（极坐标：半径，角度）
    initial_positions_polar = np.array([
        [0, 0],       # FY00 (圆心)
        [100, 0],     # FY01
        [98, 40.10],  # FY02
        [112, 80.21], # FY03
        [105, 119.75],# FY04
        [98, 159.86], # FY05
        [112, 199.96],# FY06
        [105, 240.07],# FY07
        [98, 280.17], # FY08
        [112, 320.28] # FY09
    ])
    
    # 转换为直角坐标系
    current_positions = np.array([polar_to_cartesian(r, theta) for r, theta in initial_positions_polar])
    
    # 无人机编号
    drone_ids = [f'FY{i:02d}' for i in range(10)]
    
    # 周边无人机索引（FY01-FY09）
    perimeter_indices = list(range(1, 10))
    center_index = 0  # FY00
    
    # 理想角度间隔（每40度一架）
    ideal_angles = np.array([i * 40.0 for i in range(9)])
    
    # 算法参数
    max_iterations = 15
    tolerance = 1e-3
    step_factor = 0.3  # 位置调整步长因子
    
    print('=== 无人机编队调整算法 ===')
    print('算法：迭代 + 贪心优化')
    print(f'最大迭代次数：{max_iterations}')
    print(f'收敛阈值：{tolerance}')
    print()
    
    print(f"{'迭代':<4} {'最优发射机组合':<25} {'总误差(m)':<12} {'平均误差(m)':<12}")
    print('-' * 60)
    
    error_history = []
    
    # --- 迭代调整过程 ---
    for iteration in range(1, max_iterations + 1):
        
        # 计算当前理想半径（所有周边无人机到圆心的平均距离）
        distances_to_center = [np.linalg.norm(current_positions[i] - current_positions[center_index]) 
                             for i in perimeter_indices]
        current_ideal_radius = np.mean(distances_to_center)
        
        # 生成理想圆上的位置
        ideal_positions = np.array([polar_to_cartesian(current_ideal_radius, angle) 
                                  for angle in ideal_angles])
        
        # 贪心搜索最优发射机组合
        best_total_error = float('inf')
        best_transmitter_combo = None
        best_new_positions = None
        
        # 尝试所有可能的发射机组合（FY00 + 1到3架周边无人机）
        for num_transmitters in range(1, 4):  # 1, 2, 3架周边发射机
            for transmitter_combo in combinations(perimeter_indices, num_transmitters):
                transmitter_indices = [center_index] + list(transmitter_combo)
                receiver_indices = [i for i in perimeter_indices if i not in transmitter_combo]
                
                # 复制当前位置进行模拟调整
                temp_positions = copy.deepcopy(current_positions)
                
                # 对每个接收机进行定位和调整
                for receiver_idx in receiver_indices:
                    # 测量角度（基于当前实际位置）
                    transmitter_positions = [temp_positions[i] for i in transmitter_indices]
                    
                    # 计算角度信息
                    measured_angles = []
                    if len(transmitter_positions) >= 2:
                        # FY00与第一个发射机的夹角
                        angle1 = calculate_angle_deg(transmitter_positions[0], 
                                                   temp_positions[receiver_idx],
                                                   transmitter_positions[1])
                        measured_angles.append(angle1)
                        
                        if len(transmitter_positions) >= 3:
                            # FY00与第二个发射机的夹角
                            angle2 = calculate_angle_deg(transmitter_positions[0],
                                                       temp_positions[receiver_idx],
                                                       transmitter_positions[2])
                            measured_angles.append(angle2)
                    
                    # 定位计算
                    if measured_angles:
                        new_pos = locate_drone_trilateration(temp_positions[receiver_idx],
                                                           transmitter_positions,
                                                           measured_angles)
                        
                        # 渐进式调整位置
                        temp_positions[receiver_idx] = adjust_position_gradually(
                            temp_positions[receiver_idx], new_pos, step_factor)
                        
                        # 额外的圆形约束：将无人机拉向理想圆周
                        center_pos = temp_positions[center_index]
                        current_dist = np.linalg.norm(temp_positions[receiver_idx] - center_pos)
                        if current_dist > 0:
                            # 计算在理想圆上的位置
                            direction = (temp_positions[receiver_idx] - center_pos) / current_dist
                            ideal_pos_on_circle = center_pos + direction * current_ideal_radius
                            
                            # 向理想圆位置调整
                            temp_positions[receiver_idx] = adjust_position_gradually(
                                temp_positions[receiver_idx], ideal_pos_on_circle, step_factor * 0.5)
                
                # 计算调整后的总误差
                errors = []
                for i, idx in enumerate(perimeter_indices):
                    if idx not in transmitter_combo:  # 只计算接收机的误差
                        error = np.linalg.norm(temp_positions[idx] - ideal_positions[i])
                        errors.append(error)
                
                total_error = sum(errors) if errors else float('inf')
                
                # 更新最优组合
                if total_error < best_total_error:
                    best_total_error = total_error
                    best_transmitter_combo = transmitter_indices
                    best_new_positions = temp_positions
        
        # 应用最优调整
        if best_new_positions is not None:
            current_positions = best_new_positions
            best_drone_names = [drone_ids[i] for i in best_transmitter_combo]
        else:
            best_drone_names = ["无有效组合"]
        
        # 计算当前总误差和平均误差
        current_errors = []
        for i, idx in enumerate(perimeter_indices):
            error = np.linalg.norm(current_positions[idx] - ideal_positions[i])
            current_errors.append(error)
        
        total_error = sum(current_errors)
        avg_error = np.mean(current_errors)
        error_history.append(total_error)
        
        print(f"{iteration:<4} {' '.join(best_drone_names):<25} {total_error:<12.6f} {avg_error:<12.6f}")
        
        # 检查收敛
        if total_error < tolerance:
            print(f'\n算法在第{iteration}次迭代后收敛！')
            break
    
    # --- 输出最终结果 ---
    final_positions_polar = np.array([cartesian_to_polar(pos[0], pos[1]) for pos in current_positions])
    
    # 重新计算最终理想半径
    final_distances = [np.linalg.norm(current_positions[i] - current_positions[center_index]) 
                      for i in perimeter_indices]
    final_ideal_radius = np.mean(final_distances)
    final_ideal_positions = np.array([polar_to_cartesian(final_ideal_radius, angle) 
                                    for angle in ideal_angles])
    
    # 计算最终误差
    final_errors = []
    for i, idx in enumerate(perimeter_indices):
        error = np.linalg.norm(current_positions[idx] - final_ideal_positions[i])
        final_errors.append(error)
    
    print('\n=== 调整完成 ===')
    print(f'最终理想圆半径：{final_ideal_radius:.2f} m')
    print(f'平均位置误差：{np.mean(final_errors):.6f} m')
    print(f'最大位置误差：{np.max(final_errors):.6f} m')
    
    print('\n=== 最终无人机位置 ===')
    print(f"{'编号':<6} {'初始位置(ρ,θ°)':<20} {'最终位置(ρ,θ°)':<20} {'到理想位置距离(m)':<20}")
    print('-' * 75)
    
    for i in range(10):
        initial_polar = initial_positions_polar[i]
        final_polar = final_positions_polar[i]
        
        if i == 0:  # FY00
            print(f"{drone_ids[i]:<6} ({initial_polar[0]:.2f}, {initial_polar[1]:.2f}°)"
                  f"    ({final_polar[0]:.2f}, {final_polar[1]:.2f}°)"
                  f"    {'圆心' :>15}")
        else:  # FY01-FY09
            error_distance = final_errors[i-1]
            print(f"{drone_ids[i]:<6} ({initial_polar[0]:.2f}, {initial_polar[1]:.2f}°)"
                  f"    ({final_polar[0]:.2f}, {final_polar[1]:.2f}°)"
                  f"    {error_distance:>15.6f}")
    
    # --- 可视化结果 ---
    visualize_results(initial_positions_polar, current_positions, 
                     final_ideal_radius, error_history, drone_ids)

def visualize_results(initial_positions_polar, final_positions, 
                     final_ideal_radius, error_history, drone_ids):
    """可视化调整结果"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1：初始位置
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("初始无人机位置", fontsize=14)
    
    # FY00
    ax1.scatter(0, 0, s=120, c='red', marker='o', label='FY00 (圆心)', zorder=5)
    
    # FY01-FY09初始位置
    initial_cart = [polar_to_cartesian(r, theta) for r, theta in initial_positions_polar[1:]]
    initial_x = [pos[0] for pos in initial_cart]
    initial_y = [pos[1] for pos in initial_cart]
    
    ax1.scatter(initial_x, initial_y, s=80, c='blue', marker='^', 
               label='FY01-FY09', alpha=0.7, zorder=4)
    
    # 标注无人机编号
    for i, (x, y) in enumerate(initial_cart):
        ax1.annotate(drone_ids[i+1], (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # 画理想圆
    theta = np.linspace(0, 2*np.pi, 1000)
    ideal_x = 100 * np.cos(theta)
    ideal_y = 100 * np.sin(theta)
    ax1.plot(ideal_x, ideal_y, 'k--', alpha=0.5, label='理想圆(R=100m)')
    
    ax1.set_xlim(-130, 130)
    ax1.set_ylim(-130, 130)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 子图2：最终位置
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("调整后无人机位置", fontsize=14)
    
    # FY00
    ax2.scatter(final_positions[0,0], final_positions[0,1], s=120, c='red', 
               marker='o', label='FY00 (圆心)', zorder=5)
    
    # FY01-FY09最终位置
    final_x = final_positions[1:, 0]
    final_y = final_positions[1:, 1]
    
    ax2.scatter(final_x, final_y, s=80, c='green', marker='*', 
               label='FY01-FY09 (调整后)', alpha=0.8, zorder=4)
    
    # 标注编号
    for i, (x, y) in enumerate(zip(final_x, final_y)):
        ax2.annotate(drone_ids[i+1], (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # 画最终理想圆
    final_ideal_x = final_ideal_radius * np.cos(theta)
    final_ideal_y = final_ideal_radius * np.sin(theta)
    ax2.plot(final_ideal_x, final_ideal_y, 'k--', alpha=0.5, 
            label=f'最终理想圆(R={final_ideal_radius:.1f}m)')
    
    ax2.set_xlim(-130, 130)
    ax2.set_ylim(-130, 130)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 子图3：误差收敛曲线
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("误差收敛过程", fontsize=14)
    
    iterations = range(1, len(error_history) + 1)
    ax3.plot(iterations, error_history, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.set_xlabel("迭代次数")
    ax3.set_ylabel("总误差 (m)")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 对数坐标更好显示收敛过程
    
    # 子图4：误差分布
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("各无人机位置误差", fontsize=14)
    
    # 计算每架无人机的误差
    ideal_angles = np.array([i * 40.0 for i in range(9)])
    final_ideal_positions = np.array([polar_to_cartesian(final_ideal_radius, angle) 
                                    for angle in ideal_angles])
    
    individual_errors = []
    for i in range(9):
        error = np.linalg.norm(final_positions[i+1] - final_ideal_positions[i])
        individual_errors.append(error)
    
    bars = ax4.bar(range(1, 10), individual_errors, color='lightcoral', alpha=0.7)
    ax4.set_xlabel("无人机编号")
    ax4.set_ylabel("位置误差 (m)")
    ax4.set_xticks(range(1, 10))
    ax4.set_xticklabels([f'FY{i:02d}' for i in range(1, 10)], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()