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

# 简单的距离误差计算（不优化，只计算）
def calculate_distance_error(point, points, alphas):
    """计算点到三个圆的距离误差之和"""
    try:
        residuals = total_residual(point, points, alphas)
        # 返回误差的平方和
        return np.sum(np.array(residuals)**2)
    except:
        return float('inf')

# 主函数 - 使用两阶段搜索方法
def find_intersection_two_stage(r_thetas, alphas_deg, search_radius=120, num_points_per_circle=36, num_best_points=10):
    """
    两阶段搜索方法：
    第一阶段：遍历圆周上的点，计算距离误差，找出最小的几个点
    第二阶段：对这些点进行优化
    """
    
    # 极坐标转直角坐标
    points = [polar_to_cartesian(r, theta) for r, theta in r_thetas]
    alphas = [math.radians(a) for a in alphas_deg]

    # 在多个半径的圆周上遍历点
    radii = np.linspace(20, search_radius, 6)  # 6个不同的半径
    
    all_points_errors = []
    
    for radius in radii:
        for i in range(num_points_per_circle):
            angle = 2 * np.pi * i / num_points_per_circle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # 计算距离误差
            error = calculate_distance_error([x, y], points, alphas)
            
            if error < float('inf'):
                all_points_errors.append(((x, y), error))
    
    # 按误差排序，找出最小的几个点
    all_points_errors.sort(key=lambda x: x[1])
    best_points = all_points_errors[:min(num_best_points, len(all_points_errors))]
    
    best_result = None
    best_error = float('inf')
    best_initial_point = None
    
    for point, error in best_points:
        x, y = point
        try:
            # 对这个点进行优化
            result = least_squares(total_residual, [x, y], args=(points, alphas), method='lm')
            
            # 计算优化后的总误差
            total_error = np.sum(np.array(result.fun)**2)
            
            # 如果误差更小，则更新最佳结果
            if total_error < best_error:
                best_error = total_error
                best_result = result
                best_initial_point = (x, y)
                
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

# 绘制单个测试点的图形
def plot_single_test_result(fixed_points_cartesian, fixed_points_polar, true_point, calc_point, error, test_point_index, angles, base_angle):
    """绘制单个测试点的结果，包括三个圆弧"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    true_x, true_y = true_point
    calc_x, calc_y = calc_point
    
    # 绘制参考圆（半径100）
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 100 * np.cos(circle_theta)
    circle_y = 100 * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, '--', color='lightgray', alpha=0.7, linewidth=1, label='Reference Circle (r=100)')
    
    # 绘制固定参考点
    fixed_colors = ['red', 'blue', 'green']
    fixed_labels = ['P1(Center)', 'P2(40°)', 'P3(80°)']
    
    for i, (point, color, label) in enumerate(zip(fixed_points_cartesian, fixed_colors, fixed_labels)):
        x, y = point
        ax.plot(x, y, 'o', color=color, markersize=12, label=label)
        ax.annotate(f'{label}\n({x:.1f}, {y:.1f})', 
                   (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 绘制测试点和计算点
    # 真实点
    ax.plot(true_x, true_y, 's', color='orange', markersize=12, label=f'True Position')
    ax.annotate(f'True Position\n({true_x:.2f}, {true_y:.2f})\nAngle: {base_angle:.1f}°', 
               (true_x, true_y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 计算点
    ax.plot(calc_x, calc_y, '*', color='black', markersize=15, label=f'Calculated Position')
    ax.annotate(f'Calculated Position\n({calc_x:.2f}, {calc_y:.2f})\nError: {error:.2f}', 
               (calc_x, calc_y), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 连接线（显示误差）
    ax.plot([true_x, calc_x], [true_y, calc_y], '--', color='gray', alpha=0.7, linewidth=1.5)
    
    # 绘制三个弦
    chords = [
        (fixed_points_cartesian[0], fixed_points_cartesian[1], 'P1-P2', 'red'),
        (fixed_points_cartesian[1], fixed_points_cartesian[2], 'P2-P3', 'blue'),
        (fixed_points_cartesian[2], fixed_points_cartesian[0], 'P3-P1', 'green')
    ]
    
    for i, (point_a, point_b, label, color) in enumerate(chords):
        x_coords = [point_a[0], point_b[0]]
        y_coords = [point_a[1], point_b[1]]
        ax.plot(x_coords, y_coords, '--', color=color, alpha=0.6, linewidth=1.5)
        mid_x = (point_a[0] + point_b[0]) / 2
        mid_y = (point_a[1] + point_b[1]) / 2
        ax.annotate(f'{label}\nα={angles[i]:.1f}°', 
                   (mid_x, mid_y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, color=color)
    
    # 绘制三个圆弧
    arc_info = [
        (fixed_points_cartesian[0], fixed_points_cartesian[1], math.radians(angles[0]), 'Arc P1-P2', 'red'),
        (fixed_points_cartesian[1], fixed_points_cartesian[2], math.radians(angles[1]), 'Arc P2-P3', 'blue'),
        (fixed_points_cartesian[2], fixed_points_cartesian[0], math.radians(angles[2]), 'Arc P3-P1', 'green')
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
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Three-Arc Intersection Localization - Test Point {test_point_index}\n'
                f'True Angle: {base_angle:.1f}°, Error: {error:.2f}', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# 主测试程序
def main():
    print("=== Three-Arc Intersection Localization Test (9-Sector System) ===")
    
    # 设置圆参数
    circle_radius = 100
    center_point = (0, 0)
    
    # 三个固定参考点（9等分点）
    # 9等分点角度: 0°, 40°, 80°, 120°, 160°, 200°, 240°, 280°, 320°
    # 选择其中三个作为参考点
    reference_angles_deg = [0, 40, 80]  # 参考点角度
    
    fixed_points_polar = [
        (0, math.radians(0)),                           # 圆心
        (circle_radius, math.radians(reference_angles_deg[1])),  # 40°点
        (circle_radius, math.radians(reference_angles_deg[2]))   # 80°点
    ]
    
    # 转换为直角坐标
    fixed_points_cartesian = [polar_to_cartesian(r, theta) for r, theta in fixed_points_polar]
    
    print("Fixed Reference Points:")
    for i, (point, polar, angle_deg) in enumerate(zip(fixed_points_cartesian, fixed_points_polar, [0] + reference_angles_deg[1:])):
        print(f"  Point {i+1}: Cartesian({point[0]:.2f}, {point[1]:.2f}), Polar(r={polar[0]:.1f}, θ={angle_deg:.1f}°)")
    
    # 生成测试点（在其他9等分点上，可有±1的浮动）
    np.random.seed(42)  # 固定随机种子以便复现
    
    # 所有9等分点角度
    all_9_points_angles = [i * 40 for i in range(9)]  # 0, 40, 80, 120, 160, 200, 240, 280, 320
    print(f"\nAll 9-sector angles: {all_9_points_angles}")
    
    # 排除参考点角度
    available_angles = [angle for angle in all_9_points_angles if angle not in reference_angles_deg]
    print(f"Available angles for testing: {available_angles}")
    
    # 选择几个测试角度
    selected_test_angles = available_angles[:3]  # 选择前3个可用角度进行演示
    print(f"Selected test angles: {selected_test_angles}")
    
    # 对每个测试点进行单独测试和绘图
    total_error_distance = 0
    total_error_angle = 0
    
    for test_idx, base_angle in enumerate(selected_test_angles):
        print(f"\n=== Test Point {test_idx+1} (Base Angle: {base_angle}°) ===")
        
        # 添加±10度的浮动
        angle_float = np.random.uniform(-1, 1)
        actual_angle = base_angle + angle_float
        actual_angle_rad = math.radians(actual_angle)
        
        # 在半径100的圆上生成点
        r = circle_radius
        true_x, true_y = polar_to_cartesian(r, actual_angle_rad)
        print(f"True Position: ({true_x:.2f}, {true_y:.2f})")
        print(f"Actual Angle: {actual_angle:.2f}° (Base {base_angle}° + Float {angle_float:.2f}°)")
        
        # 计算三个夹角
        angles = []
        point_labels = ['P1-P2', 'P2-P3', 'P3-P1']
        for j in range(3):
            p1_idx = j
            p2_idx = (j + 1) % 3
            angle = calculate_angle_between_points(
                (true_x, true_y), 
                fixed_points_cartesian[p1_idx], 
                fixed_points_cartesian[p2_idx]
            )
            angles.append(math.degrees(angle))
        
        print(f"Observed Angles: {point_labels[0]}={angles[0]:.2f}°, {point_labels[1]}={angles[1]:.2f}°, {point_labels[2]}={angles[2]:.2f}°")
        
        # 使用算法计算位置
        try:
            calc_r, calc_theta, calc_xy, error = find_intersection_two_stage(
                fixed_points_polar, angles, search_radius=120, num_points_per_circle=36, num_best_points=10
            )
            
            calc_x, calc_y = calc_xy
            print(f"Calculated Position: ({calc_x:.2f}, {calc_y:.2f})")
            
            # 计算误差
            distance_error = np.sqrt((true_x - calc_x)**2 + (true_y - calc_y)**2)
            angle_error = distance_error / circle_radius * 180 / np.pi  # 转换为角度误差
            
            print(f"Localization Error: {distance_error:.2f} (Distance), {angle_error:.2f}° (Angle)")
            
            total_error_distance += distance_error
            total_error_angle += angle_error
            
            # 绘制单个测试点的结果
            plot_single_test_result(
                fixed_points_cartesian, fixed_points_polar,
                (true_x, true_y), calc_xy, distance_error,
                test_idx+1, angles, actual_angle
            )
            
        except Exception as e:
            print(f"Calculation failed: {e}")
            continue
    
    print(f"\n=== Statistical Results ===")
    print(f"Number of test points: {len(selected_test_angles)}")
    print(f"Average distance error: {total_error_distance/len(selected_test_angles):.2f}")
    print(f"Average angle error: {total_error_angle/len(selected_test_angles):.2f}°")

if __name__ == "__main__":
    main()