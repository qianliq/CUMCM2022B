import numpy as np
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt

# 极坐标转直角坐标
def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

# 直角坐标转极坐标
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

# 构造圆心和半径（使用你提供的公式）
def get_circle_params(p1, p2, alpha):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    d_squared = dx**2 + dy**2
    d = np.sqrt(d_squared)

    # 半径
    sin_alpha = np.sin(alpha)
    if sin_alpha == 0:
        raise ValueError("Alpha cannot be 0 or 180 degrees.")
    radius = d / (2 * sin_alpha)

    # 中点
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # 垂直方向单位向量（绕中点旋转90度）
    cot_alpha = abs(np.cos(alpha) / np.sin(alpha))

    # 圆心偏移方向（两个方向）
    offset_x1 = -dy * cot_alpha / 2
    offset_y1 = dx * cot_alpha / 2

    offset_x2 = dy * cot_alpha / 2
    offset_y2 = -dx * cot_alpha / 2

    center1 = (mx + offset_x1, my + offset_y1)
    center2 = (mx + offset_x2, my + offset_y2)

    # 返回两个圆心和半径
    return center1, center2, radius

# 判断点是否在指定弧段上（符号判断）
def is_on_correct_arc(point, p1, p2, alpha):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    D = (x - x1) * dy - (y - y1) * dx
    cos_alpha = np.cos(alpha)
    
    # 根据alpha的大小决定取优弧还是劣弧
    if alpha < np.pi/2:  # alpha < 90°，取优弧
        return D * cos_alpha > 0
    else:  # alpha > 90°，取劣弧
        return D * cos_alpha < 0

# 生成圆弧上的点
def generate_arc_points(center, radius, p1, p2, alpha, num_points=100):
    """生成满足条件的圆弧点"""
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2
    
    # 计算圆心到两个端点的角度
    angle1 = np.arctan2(y1 - cy, x1 - cx)
    angle2 = np.arctan2(y2 - cy, x2 - cx)
    
    # 确保角度在正确范围内
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)
    
    # 处理跨越0度的情况
    if abs(angle2 - angle1) > np.pi:
        if angle1 < angle2:
            angle1 += 2 * np.pi
        else:
            angle2 += 2 * np.pi
    
    # 根据alpha决定取优弧还是劣弧
    arc_length = abs(angle2 - angle1)
    if alpha < np.pi/2:  # alpha < 90°，取优弧
        if arc_length < np.pi:
            # 需要取劣弧，所以交换角度
            angle1, angle2 = angle2, angle1
            if abs(angle2 - angle1) < np.pi:
                if angle1 < angle2:
                    angle1 += 2 * np.pi
                else:
                    angle2 += 2 * np.pi
    else:  # alpha > 90°，取劣弧
        if arc_length > np.pi:
            # 需要取优弧，所以交换角度
            angle1, angle2 = angle2, angle1
            if abs(angle2 - angle1) > np.pi:
                if angle1 < angle2:
                    angle1 += 2 * np.pi
                else:
                    angle2 += 2 * np.pi
    
    # 生成角度数组
    if angle1 < angle2:
        angles = np.linspace(angle1, angle2, num_points)
    else:
        angles = np.linspace(angle2, angle1, num_points)
    
    # 计算点坐标
    x_points = cx + radius * np.cos(angles)
    y_points = cy + radius * np.sin(angles)
    
    return x_points, y_points

# 圆的残差函数（用于最小二乘优化）
def residual(point, p1, p2, alpha):
    x, y = point
    center1, center2, radius = get_circle_params(p1, p2, alpha)

    dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
    dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)

    # 选择满足弧段条件的圆
    if is_on_correct_arc(point, p1, p2, alpha):
        # 检查哪个圆心满足弧段条件
        if is_on_correct_arc(center1, p1, p2, alpha):
            return dist1 - radius
        elif is_on_correct_arc(center2, p1, p2, alpha):
            return dist2 - radius
        else:
            return min(abs(dist1 - radius), abs(dist2 - radius))
    else:
        # 如果点不在正确弧段上，返回较大的误差
        return min(abs(dist1 - radius), abs(dist2 - radius))

# 总残差函数（三个圆的残差平方和）
def total_residual(point, points, alphas):
    p1, p2, p3 = points
    a12, a23, a31 = alphas
    r1 = residual(point, p1, p2, a12)
    r2 = residual(point, p2, p3, a23)
    r3 = residual(point, p3, p1, a31)
    return [r1, r2, r3]

# 主函数 - 从飞机对应编号的初始坐标开始搜索
def find_intersection_from_aircraft_initial(aircraft_id, fixed_points_cartesian, alphas_deg, search_radius=20, grid_step=1):
    """
    从飞机对应编号的初始坐标开始搜索交点
    """
    
    # 解析飞机编号，获取对应的初始参考点
    if aircraft_id.startswith("FY") and len(aircraft_id) == 4:
        try:
            point_index = int(aircraft_id[2:4]) - 1  # FY01->0, FY02->1, ...
            if 0 <= point_index < len(fixed_points_cartesian):
                initial_point = fixed_points_cartesian[point_index]
            else:
                # 如果编号超出范围，使用圆心作为初始点
                initial_point = fixed_points_cartesian[0]  # FY00
        except:
            # 如果解析失败，使用圆心作为初始点
            initial_point = fixed_points_cartesian[0]  # FY00
    else:
        # 如果格式不正确，使用圆心作为初始点
        initial_point = fixed_points_cartesian[0]  # FY00
    
    alphas = [math.radians(a) for a in alphas_deg]
    best_result = None
    best_error = float('inf')
    
    # 从初始点周围进行网格搜索
    initial_x, initial_y = initial_point
    
    # 在初始点周围进行网格搜索
    x_range = np.arange(initial_x - search_radius, initial_x + search_radius + grid_step, grid_step)
    y_range = np.arange(initial_y - search_radius, initial_y + search_radius + grid_step, grid_step)
    
    for x in x_range:
        for y in y_range:
            try:
                # 对这个点进行优化
                result = least_squares(total_residual, [x, y], args=(fixed_points_cartesian, alphas), method='lm')
                
                # 计算优化后的总误差
                total_error = np.sum(np.array(result.fun)**2)
                
                # 如果误差更小，则更新最佳结果
                if total_error < best_error:
                    best_error = total_error
                    best_result = result
                    
            except Exception as e:
                continue
    
    if best_result is not None:
        # 转换回极坐标
        x, y = best_result.x
        r, theta = cartesian_to_polar(x, y)
        return r, theta, (x, y), best_error
    else:
        raise ValueError("Could not find intersection point")

# 计算两点间角度（弧度）
def calculate_angle_between_points(observer, point1, point2):
    """计算观察者看两点间的夹角"""
    x_obs, y_obs = observer
    x1, y1 = point1
    x2, y2 = point2
    
    # 向量1: 从观察者到点1
    vec1_x = x1 - x_obs
    vec1_y = y1 - y_obs
    
    # 向量2: 从观察者到点2
    vec2_x = x2 - x_obs
    vec2_y = y2 - y_obs
    
    # 计算向量夹角
    dot_product = vec1_x * vec2_x + vec1_y * vec2_y
    magnitude1 = np.sqrt(vec1_x**2 + vec1_y**2)
    magnitude2 = np.sqrt(vec2_x**2 + vec2_y**2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    # 限制cos值在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.arccos(cos_angle)
    return angle

# 绘制所有点的综合图形
def plot_all_results(fixed_points_cartesian, fixed_point_names, test_results, test_reference_points_cartesian, test_reference_point_names):
    """绘制所有测试点的综合结果图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # 绘制参考圆（半径100）
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 100 * np.cos(circle_theta)
    circle_y = 100 * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, '--', color='lightgray', alpha=0.7, linewidth=1, label='Reference Circle (r=100)')
    
    # 绘制固定参考点（FY00, FY01, FY02）
    fixed_colors = ['red', 'blue', 'green']
    for i, (point, color, name) in enumerate(zip(test_reference_points_cartesian, fixed_colors, test_reference_point_names)):
        x, y = point
        ax.plot(x, y, 'o', color=color, markersize=12, label=f'{name} (Reference)')
        r, theta = cartesian_to_polar(x, y)
        ax.annotate(f'{name}\n(r={r:.1f}, θ={math.degrees(theta):.1f}°)', 
                   (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 绘制所有测试点和计算点
    true_positions_x = []
    true_positions_y = []
    calc_positions_x = []
    calc_positions_y = []
    errors = []
    aircraft_ids = []
    
    for result in test_results:
        # 真实点
        true_x, true_y = result['true_position']
        true_positions_x.append(true_x)
        true_positions_y.append(true_y)
        
        # 计算点
        calc_x, calc_y = result['calculated_position']
        calc_positions_x.append(calc_x)
        calc_positions_y.append(calc_y)
        
        # 误差和编号
        errors.append(result['distance_error'])
        aircraft_ids.append(result['aircraft_id'])
    
    # 绘制真实点
    scatter_true = ax.scatter(true_positions_x, true_positions_y, c='orange', s=80, marker='s', 
                             label='True Positions', edgecolors='black', linewidth=0.5)
    
    # 绘制计算点
    scatter_calc = ax.scatter(calc_positions_x, calc_positions_y, c='black', s=100, marker='*', 
                             label='Calculated Positions', edgecolors='white', linewidth=0.5)
    
    # 为每个点添加标注
    for i, (true_x, true_y, calc_x, calc_y, aircraft_id, error) in enumerate(zip(
        true_positions_x, true_positions_y, calc_positions_x, calc_positions_y, aircraft_ids, errors)):
        
        # 真实点标注
        r_true, theta_true = cartesian_to_polar(true_x, true_y)
        ax.annotate(f'{aircraft_id}\nT({r_true:.1f}, {math.degrees(theta_true):.1f}°)', 
                   (true_x, true_y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=7, fontweight='bold', color='orange')
        
        # 计算点标注
        r_calc, theta_calc = cartesian_to_polar(calc_x, calc_y)
        ax.annotate(f'C({r_calc:.1f}, {math.degrees(theta_calc):.1f}°)\nErr:{error:.2f}', 
                   (calc_x, calc_y), xytext=(5, -15), textcoords='offset points', 
                   fontsize=7, fontweight='bold', color='black')
        
        # 连接线（显示误差）
        ax.plot([true_x, calc_x], [true_y, calc_y], '--', color='gray', alpha=0.5, linewidth=0.8)
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Three-Arc Intersection Localization - All Test Points\n'
                'Orange squares: True Positions, Black stars: Calculated Positions', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# 绘制单个飞机的详细测量过程
def plot_single_aircraft_detail(aircraft_id, true_position, calculated_position, distance_error, 
                               test_reference_points_cartesian, test_reference_point_names, angles):
    """绘制单个飞机的详细测量过程，显示三个圆弧"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    true_x, true_y = true_position
    calc_x, calc_y = calculated_position
    
    # 绘制参考圆（半径100）
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 100 * np.cos(circle_theta)
    circle_y = 100 * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, '--', color='lightgray', alpha=0.7, linewidth=1, label='Reference Circle (r=100)')
    
    # 绘制固定参考点
    fixed_colors = ['red', 'blue', 'green']
    for i, (point, color, name) in enumerate(zip(test_reference_points_cartesian, fixed_colors, test_reference_point_names)):
        x, y = point
        ax.plot(x, y, 'o', color=color, markersize=12, label=f'{name} (Reference)')
        r, theta = cartesian_to_polar(x, y)
        ax.annotate(f'{name}\n(r={r:.1f}, θ={math.degrees(theta):.1f}°)', 
                   (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 绘制真实点和计算点
    # 真实点
    r_true, theta_true = cartesian_to_polar(true_x, true_y)
    ax.plot(true_x, true_y, 's', color='orange', markersize=12, label=f'True Position ({aircraft_id})')
    ax.annotate(f'True Position\n(r={r_true:.2f}, θ={math.degrees(theta_true):.2f}°)', 
               (true_x, true_y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 计算点
    r_calc, theta_calc = cartesian_to_polar(calc_x, calc_y)
    ax.plot(calc_x, calc_y, '*', color='black', markersize=15, label=f'Calculated Position')
    ax.annotate(f'Calculated Position\n(r={r_calc:.2f}, θ={math.degrees(theta_calc):.2f}°)\nError: {distance_error:.2f}m', 
               (calc_x, calc_y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 连接线（显示误差）
    ax.plot([true_x, calc_x], [true_y, calc_y], '--', color='gray', alpha=0.7, linewidth=1.5)
    
    # 绘制三个弦
    chord_pairs = [(0, 1), (1, 2), (2, 0)]
    chord_labels = ['FY00-FY01', 'FY01-FY02', 'FY02-FY00']
    chord_colors = ['red', 'blue', 'green']
    
    for i, ((idx1, idx2), label, color) in enumerate(zip(chord_pairs, chord_labels, chord_colors)):
        point_a = test_reference_points_cartesian[idx1]
        point_b = test_reference_points_cartesian[idx2]
        x_coords = [point_a[0], point_b[0]]
        y_coords = [point_a[1], point_b[1]]
        ax.plot(x_coords, y_coords, '--', color=color, alpha=0.6, linewidth=1.5)
        mid_x = (point_a[0] + point_b[0]) / 2
        mid_y = (point_a[1] + point_b[1]) / 2
        ax.annotate(f'{label}\nα={angles[i]:.2f}°', 
                   (mid_x, mid_y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, color=color)
    
    # 绘制三个圆弧
    arc_info = [
        (test_reference_points_cartesian[0], test_reference_points_cartesian[1], math.radians(angles[0]), 'Arc FY00-FY01', 'red'),
        (test_reference_points_cartesian[1], test_reference_points_cartesian[2], math.radians(angles[1]), 'Arc FY01-FY02', 'blue'),
        (test_reference_points_cartesian[2], test_reference_points_cartesian[0], math.radians(angles[2]), 'Arc FY02-FY00', 'green')
    ]
    
    for i, (point_a, point_b, alpha, label, color) in enumerate(arc_info):
        try:
            center1, center2, radius = get_circle_params(point_a, point_b, alpha)
            
            # 生成两个圆弧
            x_arc1, y_arc1 = generate_arc_points(center1, radius, point_a, point_b, alpha)
            x_arc2, y_arc2 = generate_arc_points(center2, radius, point_a, point_b, alpha)
            
            # 绘制两个圆弧（用不同线型区分）
            ax.plot(x_arc1, y_arc1, '-', color=color, alpha=0.7, linewidth=2, label=label if i == 0 else "")
            ax.plot(x_arc2, y_arc2, '-.', color=color, alpha=0.7, linewidth=2)
            
            # 绘制圆心
            ax.plot(center1[0], center1[1], 'x', color=color, markersize=6, alpha=0.8)
            ax.plot(center2[0], center2[1], 'x', color=color, markersize=6, alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not generate arc for {label}: {e}")
            continue
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Three-Arc Intersection Localization - Aircraft {aircraft_id}\n'
                f'True: (r={r_true:.2f}, θ={math.degrees(theta_true):.2f}°), '
                f'Calculated: (r={r_calc:.2f}, θ={math.degrees(theta_calc):.2f}°), '
                f'Error: {distance_error:.2f}m', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# 主测试程序
def main():
    print("=== Three-Arc Intersection Localization Test (All 9 Points - Polar Coordinates) ===")
    
    # 设置圆参数
    circle_radius = 100
    
    # 创建编号化的参考点系统
    # FY00: 圆心 (0, 0)
    # FY01-FY09: 圆上的9等分点
    fixed_point_names = [f"FY{i:02d}" for i in range(10)]  # FY00, FY01, FY02, ..., FY09
    
    # 9等分点角度: 0°, 40°, 80°, 120°, 160°, 200°, 240°, 280°, 320°
    angles_9_points = [i * 40 for i in range(9)]
    
    # 生成所有参考点的极坐标
    fixed_points_polar = [(0, math.radians(0))]  # FY00: 圆心
    for angle in angles_9_points:
        fixed_points_polar.append((circle_radius, math.radians(angle)))
    
    # 转换为直角坐标
    fixed_points_cartesian = [polar_to_cartesian(r, theta) for r, theta in fixed_points_polar]
    
    print("Numbered Reference Points (Polar Coordinates):")
    for i, (name, point, polar) in enumerate(zip(fixed_point_names, fixed_points_cartesian, fixed_points_polar)):
        print(f"  {name}: (r={polar[0]:.1f}, θ={math.degrees(polar[1]):.1f}°)")
    
    # 选择测试用的参考点组合（FY00, FY01, FY02）
    test_reference_indices = [0, 1, 2]  # FY00, FY01, FY02
    test_reference_points_cartesian = [fixed_points_cartesian[i] for i in test_reference_indices]
    test_reference_point_names = [fixed_point_names[i] for i in test_reference_indices]
    
    print(f"\nTest Reference Points: {test_reference_point_names}")
    
    # 生成测试飞机（所有9个点都测试）
    np.random.seed(42)  # 固定随机种子以便复现
    
    # 所有可用于测试的飞机编号（FY01-FY09）
    all_aircraft_ids = [f"FY{i:02d}" for i in range(1, 10)]
    print(f"All aircraft for testing: {all_aircraft_ids}")
    
    # 对所有飞机进行测试
    total_error_distance = 0
    total_error_angle = 0
    test_results = []
    
    print(f"\n=== Testing All Aircraft ===")
    
    for aircraft_idx, aircraft_id in enumerate(all_aircraft_ids):
        # 获取飞机编号对应的索引
        aircraft_number = int(aircraft_id[2:4])
        base_angle = angles_9_points[aircraft_number - 1]  # FY01对应索引0，FY02对应索引1，...
        
        print(f"\n--- Aircraft {aircraft_id} ---")
        print(f"Base Position: (r={circle_radius:.1f}, θ={base_angle:.1f}°)")
        
        # 添加扰动：±10米的径向扰动和±2度的角度扰动
        radial_float = np.random.uniform(-10, 10)  # ±10米
        angle_float = np.random.uniform(-2, 2)     # ±2度
        actual_radius = circle_radius + radial_float
        actual_angle = base_angle + angle_float
        actual_angle_rad = math.radians(actual_angle)
        
        # 生成飞机真实位置
        true_x, true_y = polar_to_cartesian(actual_radius, actual_angle_rad)
        true_r, true_theta = cartesian_to_polar(true_x, true_y)
        print(f"True Position: (r={true_r:.2f}, θ={math.degrees(true_theta):.2f}°)")
        print(f"Perturbation: Radius {radial_float:+.2f}m, Angle {angle_float:+.2f}°")
        
        # 显示初始参考点
        initial_x, initial_y = fixed_points_cartesian[aircraft_number]
        initial_r, initial_theta = cartesian_to_polar(initial_x, initial_y)
        print(f"Initial Reference: {aircraft_id} at (r={initial_r:.2f}, θ={math.degrees(initial_theta):.2f}°)")
        
        # 计算相对于测试参考点的三个夹角
        angles = []
        reference_pairs = [(0, 1), (1, 2), (2, 0)]
        
        for (idx1, idx2) in reference_pairs:
            angle = calculate_angle_between_points(
                (true_x, true_y), 
                test_reference_points_cartesian[idx1], 
                test_reference_points_cartesian[idx2]
            )
            angles.append(math.degrees(angle))
        
        print(f"Observed Angles: α₁={angles[0]:.2f}°, α₂={angles[1]:.2f}°, α₃={angles[2]:.2f}°")
        
        # 使用算法计算位置（从飞机对应编号的初始坐标开始搜索）
        try:
            calc_r, calc_theta, calc_xy, error = find_intersection_from_aircraft_initial(
                aircraft_id, test_reference_points_cartesian, angles, search_radius=20, grid_step=1
            )
            
            calc_x, calc_y = calc_xy
            print(f"Calculated Position: (r={calc_r:.2f}, θ={math.degrees(calc_theta):.2f}°)")
            
            # 计算误差
            distance_error = np.sqrt((true_x - calc_x)**2 + (true_y - calc_y)**2)
            angle_error = distance_error / circle_radius * 180 / np.pi  # 转换为角度误差
            
            print(f"Localization Error: {distance_error:.2f} (Distance), {angle_error:.2f}° (Angle)")
            
            total_error_distance += distance_error
            total_error_angle += angle_error
            
            test_results.append({
                'aircraft_id': aircraft_id,
                'base_position': (circle_radius, math.radians(base_angle)),
                'true_position': (true_x, true_y),
                'calculated_position': (calc_x, calc_y),
                'distance_error': distance_error,
                'angle_error': angle_error,
                'observed_angles': angles.copy()
            })
            
        except Exception as e:
            print(f"Calculation failed: {e}")
            continue
    
    print(f"\n=== Statistical Results ===")
    print(f"Number of test aircraft: {len(test_results)}")
    print(f"Average distance error: {total_error_distance/len(test_results):.2f}")
    print(f"Average angle error: {total_error_angle/len(test_results):.2f}°")
    
    # 输出详细结果表（极坐标形式）
    print(f"\n=== Detailed Results (Polar Coordinates) ===")
    print(f"{'Aircraft':<8} {'Base':<15} {'True':<15} {'Calculated':<15} {'Dist Err':<10} {'Angle Err':<10}")
    print(f"{'':<8} {'(r,θ°)':<15} {'(r,θ°)':<15} {'(r,θ°)':<15} {'(m)':<10} {'(°)':<10}")
    print("-" * 85)
    
    for result in test_results:
        base_r, base_theta = result['base_position']
        true_x, true_y = result['true_position']
        calc_x, calc_y = result['calculated_position']
        true_r, true_theta = cartesian_to_polar(true_x, true_y)
        calc_r, calc_theta = cartesian_to_polar(calc_x, calc_y)
        
        print(f"{result['aircraft_id']:<8} "
              f"({base_r:5.1f},{math.degrees(base_theta):5.1f}°) "
              f"({true_r:5.2f},{math.degrees(true_theta):5.2f}°) "
              f"({calc_r:5.2f},{math.degrees(calc_theta):5.2f}°) "
              f"{result['distance_error']:<10.2f} "
              f"{result['angle_error']:<10.2f}°")
    
    # 计算最大和最小误差
    if test_results:
        max_dist_error = max([r['distance_error'] for r in test_results])
        min_dist_error = min([r['distance_error'] for r in test_results])
        max_angle_error = max([r['angle_error'] for r in test_results])
        min_angle_error = min([r['angle_error'] for r in test_results])
        
        print(f"\n=== Error Range ===")
        print(f"Distance error range: {min_dist_error:.2f} - {max_dist_error:.2f} m")
        print(f"Angle error range: {min_angle_error:.2f}° - {max_angle_error:.2f}°")
    
    # 绘制所有点的综合图形
    print(f"\n=== Displaying Summary Plot ===")
    plot_all_results(fixed_points_cartesian, fixed_point_names, test_results, 
                    test_reference_points_cartesian, test_reference_point_names)
    
    # 依次显示每个飞机的详细测量过程
    print(f"\n=== Displaying Detailed Plots for Each Aircraft ===")
    for result in test_results:
        print(f"Displaying detailed plot for {result['aircraft_id']}...")
        plot_single_aircraft_detail(
            result['aircraft_id'],
            result['true_position'],
            result['calculated_position'],
            result['distance_error'],
            test_reference_points_cartesian,
            test_reference_point_names,
            result['observed_angles']
        )

if __name__ == "__main__":
    main()