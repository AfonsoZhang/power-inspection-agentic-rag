"""航线几何：距离公式与 2-opt 改良"""
import pytest

from src.mission.geo import haversine_km, nearest_neighbor_order, optimize_route, path_length_km


def test_haversine_zero_distance():
    assert haversine_km((36.65, 116.91), (36.65, 116.91)) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_latitude_is_about_111km():
    # 纬度 1 度 ≈ 111.19 km，与实现无关的独立事实，用来验证公式没写反
    assert haversine_km((36.0, 116.0), (37.0, 116.0)) == pytest.approx(111.19, abs=0.5)


def test_haversine_is_symmetric():
    a, b = (36.0823, 120.3712), (36.0721, 120.4012)
    assert haversine_km(a, b) == pytest.approx(haversine_km(b, a))


def test_nearest_neighbor_visits_every_point_once():
    points = [(36.0, 116.0), (36.1, 116.0), (36.2, 116.0), (36.05, 116.05)]
    order = nearest_neighbor_order(points)
    assert sorted(order) == list(range(len(points)))


def test_two_opt_never_worse_than_input_order():
    # 刻意构造的"交叉"顺序：0 -> 2 -> 1 -> 3 沿一条直线来回折返
    points = [(36.00, 116.0), (36.01, 116.0), (36.02, 116.0), (36.03, 116.0)]
    naive = path_length_km([points[0], points[2], points[1], points[3]])
    order, optimized = optimize_route(points)
    assert optimized <= naive + 1e-9
    assert sorted(order) == list(range(len(points)))


def test_optimize_route_handles_empty_and_single():
    assert optimize_route([]) == ([], 0.0)
    order, dist = optimize_route([(36.0, 116.0)])
    assert order == [0] and dist == 0.0
