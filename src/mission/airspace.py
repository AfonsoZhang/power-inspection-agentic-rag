"""飞行前合规校验：空域限制 + 气象窗口 + 带电体安全距离

同样是**确定性**判定层。三类判据各自独立给出结论，最后取最严的一条作为总裁决：
    forbidden  禁飞（净空区 / 禁飞区 / 气象超限）
    restricted 可飞但需降高或加限制（限高区）
    allowed    放行

数据源 data/airspace/constraints.json 是合成演示数据，真实作业以民航局 UOM 与
属地管理部门发布为准——判定逻辑可复用，数据必须替换。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from ..config import load_config
from ..ingestion.text_loader import load_assets
from .geo import haversine_km

VERDICT_RANK = {"allowed": 0, "restricted": 1, "forbidden": 2}


@dataclass
class ZoneHit:
    zone_id: str
    name: str
    category: str
    distance_km: float
    max_agl_m: float
    note: str


@dataclass
class AssetClearance:
    asset_id: str
    verdict: str                       # allowed | restricted | forbidden
    allowed_agl_m: float               # 该塔位允许的最大真高（0 表示禁飞）
    requested_agl_m: float
    zones: list[ZoneHit] = field(default_factory=list)
    emi_min_distance_m: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "zones": [asdict(z) for z in self.zones]}


@dataclass
class WeatherVerdict:
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClearanceReport:
    verdict: str
    requested_agl_m: float
    weather: WeatherVerdict
    assets: list[AssetClearance]
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "requested_agl_m": self.requested_agl_m,
            "weather": self.weather.to_dict(),
            "assets": [a.to_dict() for a in self.assets],
            "recommendations": self.recommendations,
        }


@lru_cache(maxsize=1)
def load_constraints() -> dict:
    path = load_config()["_paths"]["airspace_constraints"]
    return json.loads(path.read_text(encoding="utf-8"))


def check_weather(weather: dict | None) -> WeatherVerdict:
    """weather 形如 {"condition": "晴", "wind_mps": 6.5, "gust_mps": 9, "visibility_km": 10, "temperature_c": 12}"""
    if not weather:
        return WeatherVerdict("unknown", ["未提供气象数据，起飞前须自行确认风速/能见度/天气现象"])

    limits = load_constraints()["weather_limits"]
    reasons: list[str] = []

    condition = str(weather.get("condition", "")).strip()
    for bad in limits["prohibited_conditions"]:
        if bad and bad in condition:
            reasons.append(f"天气现象「{condition}」命中禁飞条件「{bad}」")

    checks = [
        ("wind_mps", limits["max_wind_mps"], "平均风速", "m/s", True),
        ("gust_mps", limits["max_gust_mps"], "阵风", "m/s", True),
    ]
    for key, limit, label, unit, is_max in checks:
        val = weather.get(key)
        if val is not None and is_max and float(val) > limit:
            reasons.append(f"{label} {val} {unit} 超过上限 {limit} {unit}")

    vis = weather.get("visibility_km")
    if vis is not None and float(vis) < limits["min_visibility_km"]:
        reasons.append(f"能见度 {vis} km 低于下限 {limits['min_visibility_km']} km")

    temp = weather.get("temperature_c")
    if temp is not None and float(temp) < limits["min_temperature_c"]:
        reasons.append(f"气温 {temp} ℃ 低于下限 {limits['min_temperature_c']} ℃（电池性能衰减）")

    return WeatherVerdict("forbidden" if reasons else "allowed", reasons)


def check_asset(asset_id: str, requested_agl_m: float) -> AssetClearance:
    assets = load_assets()
    asset = assets.get(asset_id)
    if not asset:
        return AssetClearance(
            asset_id=asset_id, verdict="forbidden", allowed_agl_m=0.0,
            requested_agl_m=requested_agl_m, reasons=[f"资产 {asset_id} 不在档案库中，无法校验塔位空域"],
        )

    constraints = load_constraints()
    point = (asset["location"]["lat"], asset["location"]["lon"])

    hits: list[ZoneHit] = []
    allowed_agl = requested_agl_m
    reasons: list[str] = []
    for z in constraints["zones"]:
        if z["shape"] != "circle":
            continue
        dist = haversine_km(point, (z["center"]["lat"], z["center"]["lon"]))
        if dist > z["radius_km"]:
            continue
        hits.append(ZoneHit(z["zone_id"], z["name"], z["category"], round(dist, 3),
                            float(z["max_agl_m"]), z["note"]))
        allowed_agl = min(allowed_agl, float(z["max_agl_m"]))
        reasons.append(f"落入 {z['name']}（{z['zone_id']}，距中心 {dist:.2f} km）：{z['note']}")

    if allowed_agl <= 0:
        verdict = "forbidden"
    elif allowed_agl < requested_agl_m:
        verdict = "restricted"
        reasons.append(f"申请真高 {requested_agl_m:.0f} m 超出区域限高，须降至 {allowed_agl:.0f} m 以下")
    else:
        verdict = "allowed"

    emi = float(constraints["emi_safe_distance_m"].get(str(asset["voltage_kv"]), 8.5))
    reasons.append(f"{asset['voltage_kv']} kV 带电导线最小安全接近距离 {emi} m，绕飞航迹须外扩该距离")

    return AssetClearance(
        asset_id=asset_id, verdict=verdict, allowed_agl_m=max(0.0, allowed_agl),
        requested_agl_m=requested_agl_m, zones=hits, emi_min_distance_m=emi, reasons=reasons,
    )


def check_flight(asset_ids: list[str], *, agl_m: float | None = None,
                 weather: dict | None = None) -> ClearanceReport:
    constraints = load_constraints()
    requested = float(agl_m if agl_m is not None else constraints["default_agl_m"])

    weather_verdict = check_weather(weather)
    assets = [check_asset(aid, requested) for aid in asset_ids]

    worst = max(
        [VERDICT_RANK.get(a.verdict, 0) for a in assets]
        + [VERDICT_RANK.get(weather_verdict.verdict, 0)],
        default=0,
    )
    verdict = {0: "allowed", 1: "restricted", 2: "forbidden"}[worst]

    recs: list[str] = []
    if weather_verdict.verdict == "forbidden":
        recs.append("气象条件不满足，建议改期至风速与能见度回到限值内的窗口再执行。")
    if weather_verdict.verdict == "unknown":
        recs.append("补充起飞前气象实况（风速、阵风、能见度、天气现象），否则不得放飞。")

    blocked = [a for a in assets if a.verdict == "forbidden"]
    if blocked:
        recs.append(
            "以下塔位禁飞，需改为人工登塔或申请空域许可后另行安排："
            + "、".join(a.asset_id for a in blocked)
        )
    limited = [a for a in assets if a.verdict == "restricted"]
    if limited:
        lowest = min(a.allowed_agl_m for a in limited)
        recs.append(f"限高塔位共 {len(limited)} 基，本架次作业真高须整体压到 {lowest:.0f} m 以下。")
    if verdict == "allowed":
        recs.append("空域与气象均满足，可按计划放飞；仍须保持对带电体的最小安全距离。")

    return ClearanceReport(verdict, requested, weather_verdict, assets, recs)


def format_report(report: ClearanceReport) -> str:
    label = {"allowed": "放行", "restricted": "有条件放行", "forbidden": "禁止起飞"}[report.verdict]
    lines = [
        f"## 飞行前合规校验：{label}",
        f"申请作业真高 {report.requested_agl_m:.0f} m｜校验塔位 {len(report.assets)} 基",
        "",
        f"### 气象：{report.weather.verdict}",
    ]
    lines += [f"- {r}" for r in report.weather.reasons] or ["- 满足全部气象限值"]

    lines += ["", "### 塔位逐基结论"]
    for a in report.assets:
        zone_txt = "、".join(z.zone_id for z in a.zones) or "无空域限制区命中"
        lines.append(
            f"- {a.asset_id}：{a.verdict}｜允许真高 {a.allowed_agl_m:.0f} m｜{zone_txt}"
        )

    lines += ["", "### 处置建议"] + [f"- {r}" for r in report.recommendations]
    lines.append("")
    lines.append("> 空域数据为合成演示数据，真实作业须以民航局 UOM 与属地管理部门发布为准。")
    return "\n".join(lines)
