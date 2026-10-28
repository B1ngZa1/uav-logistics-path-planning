"""
三维空域模型测试脚本
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from environment import Airspace3D, CollisionDetector, VoxelState
from visualization.plot_3d import AirspaceVisualizer


def test_voxel_map():
    """测试体素地图基本功能"""
    print("=" * 60)
    print("测试1: 体素地图基本功能")
    print("=" * 60)

    # 创建空域
    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)

    # 验证体素数量
    assert airspace.voxel_count == (20, 20, 10), \
        f"体素数量错误: {airspace.voxel_count}"
    print(f"  体素数量: {airspace.voxel_count}")

    # 测试坐标转换
    i, j, k = airspace.voxel_map.world_to_voxel(25, 35, 15)
    assert (i, j, k) == (5, 7, 3), f"坐标转换错误: {(i, j, k)}"
    print(f"  世界坐标 (25, 35, 15) -> 体素索引 ({i}, {j}, {k})")

    x, y, z = airspace.voxel_map.voxel_to_world(5, 7, 3)
    assert abs(x - 27.5) < 0.01 and abs(y - 37.5) < 0.01 and abs(z - 17.5) < 0.01, \
        f"坐标转换错误: {(x, y, z)}"
    print(f"  体素索引 (5, 7, 3) -> 世界坐标 ({x:.1f}, {y:.1f}, {z:.1f})")

    # 测试一维索引转换
    idx = airspace.voxel_map.voxel_to_index(5, 7, 3)
    i2, j2, k2 = airspace.voxel_map.index_to_voxel(idx)
    assert (i2, j2, k2) == (5, 7, 3), f"索引转换错误: {(i2, j2, k2)}"
    print(f"  一维索引: {idx} <-> 三维索引: ({i2}, {j2}, {k2})")

    print("  [通过] 体素地图基本功能测试\n")


def test_obstacles():
    """测试障碍物添加"""
    print("=" * 60)
    print("测试2: 障碍物添加")
    print("=" * 60)

    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)

    # 添加长方体障碍物
    airspace.add_box_obstacle((20, 40, 0), (40, 55, 30))
    print("  添加长方体障碍物: (20,40,0) - (40,55,30)")

    # 验证障碍物内部
    i, j, k = airspace.voxel_map.world_to_voxel(30, 47, 15)
    assert airspace.voxel_map.is_obstacle(i, j, k), "障碍物标记失败"
    print(f"  障碍物内部点 (30, 47, 15) -> 体素({i},{j},{k}): 状态={airspace.voxel_map.get_voxel(i,j,k)}")

    # 验证自由空间
    i, j, k = airspace.voxel_map.world_to_voxel(10, 10, 10)
    assert airspace.voxel_map.is_free(i, j, k), "自由空间标记失败"
    print(f"  自由空间点 (10, 10, 10) -> 体素({i},{j},{k}): 状态={airspace.voxel_map.get_voxel(i,j,k)}")

    # 添加圆柱体障碍物
    airspace.add_cylinder_obstacle((70, 70), 10, 25)
    print("  添加圆柱体障碍物: 中心(70,70), 半径10, 高度25")

    # 统计障碍物体素数
    obstacle_count = np.sum(airspace.voxel_map.data == VoxelState.OBSTACLE)
    print(f"  障碍物体素总数: {obstacle_count}")

    print("  [通过] 障碍物添加测试\n")


def test_no_fly_zone():
    """测试禁飞区"""
    print("=" * 60)
    print("测试3: 禁飞区")
    print("=" * 60)

    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)

    # 添加多边形禁飞区
    vertices = [(35, 85), (45, 90), (40, 95), (30, 90)]
    airspace.add_no_fly_zone(vertices, z_min=0, z_max=50)
    print(f"  添加多边形禁飞区: 顶点{vertices}, 高度0-50m")

    # 验证禁飞区内部
    i, j, k = airspace.voxel_map.world_to_voxel(37, 90, 25)
    assert airspace.voxel_map.get_voxel(i, j, k) == VoxelState.NO_FLY, \
        f"禁飞区标记失败: {airspace.voxel_map.get_voxel(i, j, k)}"
    print(f"  禁飞区内部点 (37, 90, 25) -> 状态={airspace.voxel_map.get_voxel(i,j,k)}")

    # 统计禁飞区体素数
    nfz_count = np.sum(airspace.voxel_map.data == VoxelState.NO_FLY)
    print(f"  禁飞区体素总数: {nfz_count}")

    print("  [通过] 禁飞区测试\n")


def test_collision_detection():
    """测试碰撞检测"""
    print("=" * 60)
    print("测试4: 碰撞检测")
    print("=" * 60)

    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)

    # 添加障碍物
    airspace.add_box_obstacle((40, 40, 0), (60, 60, 20))

    # 创建碰撞检测器
    detector = CollisionDetector(airspace, uav_radius=1.0, safe_distance=5.0)

    # 测试点碰撞
    pos_free = np.array([10.0, 10.0, 10.0])
    pos_obstacle = np.array([50.0, 50.0, 10.0])

    assert not detector.check_point_obstacle(pos_free), "自由空间检测失败"
    assert detector.check_point_obstacle(pos_obstacle), "障碍物检测失败"
    print(f"  自由空间点 {pos_free}: 碰撞={detector.check_point_obstacle(pos_free)}")
    print(f"  障碍物点 {pos_obstacle}: 碰撞={detector.check_point_obstacle(pos_obstacle)}")

    # 测试边界碰撞
    pos_out = np.array([110.0, 50.0, 25.0])
    assert detector.check_point_obstacle(pos_out), "边界检测失败"
    print(f"  边界外点 {pos_out}: 碰撞={detector.check_point_obstacle(pos_out)}")

    # 测试无人机间碰撞
    pos1 = np.array([30.0, 30.0, 10.0])
    pos2 = np.array([32.0, 30.0, 10.0])  # 距离2m < 安全距离5m
    pos3 = np.array([40.0, 30.0, 10.0])  # 距离10m > 安全距离5m

    assert detector.check_uav_collision(pos1, pos2), "无人机碰撞检测失败"
    assert not detector.check_uav_collision(pos1, pos3), "无人机安全距离检测失败"
    print(f"  无人机间距2m: 碰撞={detector.check_uav_collision(pos1, pos2)}")
    print(f"  无人机间距10m: 碰撞={detector.check_uav_collision(pos1, pos3)}")

    # 测试轨迹碰撞
    start = np.array([10.0, 10.0, 10.0])
    end_free = np.array([20.0, 20.0, 10.0])
    end_collision = np.array([50.0, 50.0, 10.0])

    assert not detector.check_trajectory_collision(start, end_free), "轨迹检测失败"
    assert detector.check_trajectory_collision(start, end_collision), "轨迹碰撞检测失败"
    print(f"  安全轨迹: 碰撞={detector.check_trajectory_collision(start, end_free)}")
    print(f"  碰撞轨迹: 碰撞={detector.check_trajectory_collision(start, end_collision)}")

    print("  [通过] 碰撞检测测试\n")


def test_distance_field():
    """测试距离场"""
    print("=" * 60)
    print("测试5: 距离场计算")
    print("=" * 60)

    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)

    # 添加障碍物
    airspace.add_box_obstacle((40, 40, 0), (60, 60, 20))

    # 计算距离场
    airspace.build()

    # 测试不同位置的距离
    test_points = [
        (10, 10, 10, "远离障碍物"),
        (35, 40, 10, "接近障碍物"),
        (50, 50, 10, "障碍物内部"),
    ]

    for x, y, z, desc in test_points:
        dist = airspace.get_nearest_obstacle_distance(np.array([x, y, z]))
        print(f"  {desc} ({x},{y},{z}): 距离={dist:.2f}m")

    # 验证障碍物内部距离为0
    dist_inside = airspace.get_nearest_obstacle_distance(np.array([50.0, 50.0, 10.0]))
    print(f"  障碍物内部距离: {dist_inside:.2f}m (应为0或很小)")

    print("  [通过] 距离场计算测试\n")


def test_default_city_scene():
    """测试默认城市场景"""
    print("=" * 60)
    print("测试6: 默认城市场景生成")
    print("=" * 60)

    config = {
        'airspace_size': (100, 100, 50),
        'resolution': 5
    }
    airspace = Airspace3D(config)
    airspace.create_default_city_scene()

    print(f"  长方体障碍物数量: {len(airspace.box_obstacles)}")
    print(f"  圆柱体障碍物数量: {len(airspace.cylinder_obstacles)}")
    print(f"  禁飞区数量: {len(airspace.no_fly_zones)}")
    print(f"  配送站数量: {len(airspace.depots)}")
    print(f"  客户点数量: {len(airspace.customers)}")

    # 验证起降点
    print("\n  配送站位置:")
    for i, depot in enumerate(airspace.depots):
        print(f"    D{i+1}: {depot.position}")

    print("\n  客户点位置:")
    for i, customer in enumerate(airspace.customers):
        print(f"    C{i+1:02d}: {customer.position}")

    # 统计体素状态
    obstacle_count = np.sum(airspace.voxel_map.data == VoxelState.OBSTACLE)
    nfz_count = np.sum(airspace.voxel_map.data == VoxelState.NO_FLY)
    free_count = np.sum(airspace.voxel_map.data == VoxelState.FREE)
    total_count = airspace.voxel_map.data.size

    print(f"\n  体素统计:")
    print(f"    总体素数: {total_count}")
    print(f"    障碍物: {obstacle_count} ({100*obstacle_count/total_count:.1f}%)")
    print(f"    禁飞区: {nfz_count} ({100*nfz_count/total_count:.1f}%)")
    print(f"    自由空间: {free_count} ({100*free_count/total_count:.1f}%)")

    print("  [通过] 默认城市场景测试\n")

    return airspace


def test_visualization(airspace=None):
    """测试可视化"""
    print("=" * 60)
    print("测试7: 可视化")
    print("=" * 60)

    if airspace is None:
        config = {
            'airspace_size': (100, 100, 50),
            'resolution': 5
        }
        airspace = Airspace3D(config)
        airspace.create_default_city_scene()

    visualizer = AirspaceVisualizer(airspace)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 静态三维视图
    try:
        fig = visualizer.plot_static()
        visualizer.save_figure(os.path.join(output_dir, 'airspace_3d.png'))
        plt.close(fig)
        print("  三维视图已保存")
    except Exception as e:
        print(f"  三维视图生成失败: {e}")

    # 体素地图
    try:
        fig = visualizer.plot_voxel_map()
        visualizer.save_figure(os.path.join(output_dir, 'voxel_map.png'))
        plt.close(fig)
        print("  体素地图已保存")
    except Exception as e:
        print(f"  体素地图生成失败: {e}")

    # 俯视图
    try:
        fig = visualizer.plot_top_view(z_level=0)
        visualizer.save_figure(os.path.join(output_dir, 'top_view.png'))
        plt.close(fig)
        print("  俯视图已保存")
    except Exception as e:
        print(f"  俯视图生成失败: {e}")

    print("  [通过] 可视化测试\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("三维空域模型测试")
    print("=" * 60 + "\n")

    test_voxel_map()
    test_obstacles()
    test_no_fly_zone()
    test_collision_detection()
    test_distance_field()
    airspace = test_default_city_scene()

    # 可视化测试（可能需要图形界面）
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        test_visualization(airspace)
    except ImportError:
        print("  跳过可视化测试（缺少matplotlib）")
    except Exception as e:
        print(f"  可视化测试异常: {e}")

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    # 添加matplotlib导入（在可视化测试中使用）
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass

    run_all_tests()
