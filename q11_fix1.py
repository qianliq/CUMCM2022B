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

# 总残差函数（三个圆的残差绝对值之和）
def total_residual(point, points, alphas):
    p1, p2, p3 = points
    a12, a23, a31 = alphas
    r1 = residual(point, p1, p2, a12)
    r2 = residual(point, p2, p3, a23)
    r3 = residual(point, p3, p1, a31)
    return abs(r1) + abs(r2) + abs(r3)  # 返回绝对值之和

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

# 使用扩散圆方法寻找交点
def find_intersection_by_diffusion(initial_point, fixed_points_cartesian, alphas_deg, max_radius=100, step_size=0.01, angle_step=0.5):
    """
    使用扩散圆方法寻找交点
    从初始点为中心，一定步长往外扩散圆来取点
    步长缩小100倍：step_size从1改为0.01，angle_step从5改为0.5
    """
    
    initial_x, initial_y = initial_point
    alphas = [math.radians(a) for a in alphas_deg]
    
    # 存储所有采样点的距离误差
    sample_radii = []
    sample_distances = []
    sample_points = []
    
    print(f"Starting diffusion search from ({initial_x:.2f}, {initial_y:.2f})")
    print(f"Max radius: {max_radius}, Step size: {step_size}")
    
    # 从中心开始向外扩散
    for radius in np.arange(0, max_radius + step_size, step_size):
        # 在当前半径的圆上均匀采样点
        num_points = max(8, int(2 * np.pi * radius / angle_step))  # 至少8个点，角度步长控制密度
        if radius == 0:
            # 中心点特殊处理
            x, y = initial_x, initial_y
            distance_error = total_residual([x, y], fixed_points_cartesian, alphas)
            sample_radii.append(0)
            sample_distances.append(distance_error)
            sample_points.append((x, y))
        else:
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = initial_x + radius * np.cos(angle)
                y = initial_y + radius * np.sin(angle)
                
                # 计算到三个圆的距离误差
                distance_error = total_residual([x, y], fixed_points_cartesian, alphas)
                
                sample_radii.append(radius)
                sample_distances.append(distance_error)
                sample_points.append((x, y))
        
        # 显示进度
        if int(radius * 100) % 1000 == 0:  # 每10个单位显示一次进度
            print(f"  Processed radius: {radius:.2f}")
    
    print(f"Sampled {len(sample_radii)} points")
    
    # 寻找极小值点
    local_minima_points = []
    local_minima_errors = []
    local_minima_radii = []
    
    # 简单的极小值检测：比相邻点都小的点
    for i in range(1, len(sample_distances) - 1):
        if (sample_distances[i] < sample_distances[i-1] and 
            sample_distances[i] < sample_distances[i+1] and
            sample_distances[i] < 5):  # 只考虑误差小于5的点
            local_minima_points.append(sample_points[i])
            local_minima_errors.append(sample_distances[i])
            local_minima_radii.append(sample_radii[i])
    
    # 如果没有找到极小值，取全局最小值
    if len(local_minima_points) == 0:
        min_idx = np.argmin(sample_distances)
        local_minima_points.append(sample_points[min_idx])
        local_minima_errors.append(sample_distances[min_idx])
        local_minima_radii.append(sample_radii[min_idx])
    
    print(f"Found {len(local_minima_points)} local minima")
    
    # 绘制半径-距离图
    plot_radius_distance_graph(sample_radii, sample_distances, local_minima_radii, local_minima_errors, initial_point)
    
    # 选择最合适的极小值点
    best_point = None
    best_error = float('inf')
    
    for i, (point, error, radius) in enumerate(zip(local_minima_points, local_minima_errors, local_minima_radii)):
        point_x, point_y = point
        # 计算该点到初始点的距离
        distance_from_initial = np.sqrt((point_x - initial_x)**2 + (point_y - initial_y)**2)
        
        print(f"Local minimum {i+1}: Point({point_x:.2f}, {point_y:.2f}), Error={error:.4f}, Distance from initial={distance_from_initial:.2f}")
        
        # 如果距离初始点不太远，且误差更小，则选择
        if distance_from_initial <= max_radius and error < best_error:
            best_error = error
            best_point = point
    
    # 如果所有极小值点都离初始点太远，返回初始点
    if best_point is None:
        print("All local minima are too far from initial point, returning initial point")
        best_point = initial_point
        best_error = total_residual(list(initial_point), fixed_points_cartesian, alphas)
    
    return best_point[0], best_point[1], best_error, (sample_radii, sample_distances)

# 绘制半径-距离图
def plot_radius_distance_graph(sample_radii, sample_distances, local_minima_radii, local_minima_errors, initial_point):
    """绘制半径-距离图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制所有采样点
    ax.scatter(sample_radii, sample_distances, c='blue', s=1, alpha=0.6, label='Sample Points')
    
    # 绘制极小值点
    if len(local_minima_radii) > 0:
        ax.scatter(local_minima_radii, local_minima_errors, c='red', s=50, marker='*', 
                  label='Local Minima', edgecolors='black', linewidth=0.5)
    
    # 标注极小值点
    for i, (radius, error) in enumerate(zip(local_minima_radii, local_minima_errors)):
        ax.annotate(f'Min {i+1}\n({radius:.1f}, {error:.3f})', 
                   (radius, error), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, fontweight='bold', color='red')
    
    ax.set_xlabel('Distance from Initial Point (m)', fontsize=12)
    ax.set_ylabel('Total Distance Error to Three Curves', fontsize=12)
    ax.set_title(f'Radius vs Distance Error\nInitial Point: ({initial_point[0]:.2f}, {initial_point[1]:.2f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

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
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
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
    print("=== Three-Arc Intersection Localization Test (Diffusion Method - Fine Resolution) ===")
    
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
        
        # 使用扩散圆方法计算位置（步长缩小100倍）
        try:
            calc_x, calc_y, error, (sample_radii, sample_distances) = find_intersection_by_diffusion(
                (initial_x, initial_y), test_reference_points_cartesian, angles, 
                max_radius=50, step_size=0.01, angle_step=0.5  # 步长缩小100倍
            )
            
            calc_r, calc_theta = cartesian_to_polar(calc_x, calc_y)
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