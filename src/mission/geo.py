"""地理与航线几何工具（纯函数，无外部依赖，可直接单测）"""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088

Point = tuple[float, float]  # (lat, lon)


def haversine_km(a: Point, b: Point) -> float:
    """两点大圆距离（km）。杆塔间距在百米量级，平面近似误差可忽略，但仍用球面公式保证跨线路时正确。"""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def path_length_km(points: list[Point]) -> float:
    return sum(haversine_km(points[i], points[i + 1]) for i in range(len(points) - 1))


def nearest_neighbor_order(points: list[Point], start: int = 0) -> list[int]:
    """最近邻构造初始访问顺序，返回下标序列。"""
    n = len(points)
    if n <= 2:
        return list(range(n))
    unvisited = set(range(n)) - {start}
    order = [start]
    while unvisited:
        cur = points[order[-1]]
        nxt = min(unvisited, key=lambda i: haversine_km(cur, points[i]))
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def two_opt(points: list[Point], order: list[int], max_passes: int = 20) -> list[int]:
    """2-opt 局部改良开放路径（不回起点）。杆塔量级在百以内，O(n²) 每轮完全够用。"""
    if len(order) < 4:
        return order

    def seg_len(seq: list[int]) -> float:
        return path_length_km([points[i] for i in seq])

    best = order[:]
    best_len = seg_len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                cand_len = seg_len(cand)
                if cand_len < best_len - 1e-9:
                    best, best_len, improved = cand, cand_len, True
        if not improved:
            break
    return best


def optimize_route(points: list[Point], start: int = 0) -> tuple[list[int], float]:
    """最近邻 + 2-opt，返回 (访问顺序, 总航程 km)。"""
    if not points:
        return [], 0.0
    order = two_opt(points, nearest_neighbor_order(points, start=start))
    return order, path_length_km([points[i] for i in order])
