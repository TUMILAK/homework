"""当地天气查询（Open-Meteo 地理编码 + 预报，无需 API Key）。"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import httpx

# WMO 天气现象代码 → 中文简述（Open-Meteo 使用）
_WMO_ZH = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "大阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _wmo_label(code: Optional[int]) -> str:
    if code is None:
        return "未知"
    return _WMO_ZH.get(int(code), f"代码{code}")


def _parse_lat_lon(location: str) -> Optional[Tuple[float, float]]:
    """支持 '23.13,113.26' 或 '23.13 113.26'。"""
    s = (location or "").strip().replace("，", ",")
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
    else:
        parts = s.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


async def _geocode_city(city: str) -> dict[str, Any]:
    name = (city or "").strip()
    if not name:
        raise ValueError("请提供城市名，例如：广州、北京、Shanghai")
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": name,
                "count": 5,
                "language": "zh",
                "format": "json",
            },
        )
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"未找到与「{name}」匹配的城市，请换更具体的地名（可加省/国）")
    hit = results[0]
    return {
        "name": hit.get("name") or name,
        "country": hit.get("country") or "",
        "admin1": hit.get("admin1") or "",
        "latitude": float(hit["latitude"]),
        "longitude": float(hit["longitude"]),
        "timezone": hit.get("timezone") or "auto",
    }


async def _fetch_forecast(
    lat: float,
    lon: float,
    *,
    timezone: str,
    forecast_days: int,
) -> dict[str, Any]:
    days = max(1, min(16, int(forecast_days)))
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": days,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]
        ),
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
        return r.json()


def _place_label(geo: dict[str, Any]) -> str:
    parts = [geo.get("name"), geo.get("admin1"), geo.get("country")]
    return " · ".join(p for p in parts if p)


def _format_current(data: dict[str, Any], place: str) -> str:
    cur = data.get("current") or {}
    lines = [
        f"地点：{place}",
        f"观测时间（当地）：{cur.get('time', '—')}",
        f"天气：{_wmo_label(cur.get('weather_code'))}",
        f"气温：{cur.get('temperature_2m', '—')} °C（体感 {cur.get('apparent_temperature', '—')} °C）",
        f"相对湿度：{cur.get('relative_humidity_2m', '—')} %",
        f"降水：{cur.get('precipitation', '—')} mm",
        f"风速：{cur.get('wind_speed_10m', '—')} km/h，风向 {cur.get('wind_direction_10m', '—')}°",
        f"坐标：{data.get('latitude')}, {data.get('longitude')}",
    ]
    return "\n".join(lines)


def _format_forecast(data: dict[str, Any], place: str, days: int) -> str:
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_sum") or []
    lines = [f"地点：{place}", f"未来 {min(days, len(times))} 天预报：", ""]
    for i, t in enumerate(times[:days]):
        w = _wmo_label(codes[i] if i < len(codes) else None)
        hi = tmax[i] if i < len(tmax) else "—"
        lo = tmin[i] if i < len(tmin) else "—"
        pr = precip[i] if i < len(precip) else "—"
        lines.append(f"- {t}：{w}，{lo}~{hi} °C，降水 {pr} mm")
    cur = data.get("current") or {}
    if cur:
        lines.extend(
            [
                "",
                "【当前实况】",
                f"{_wmo_label(cur.get('weather_code'))}，{cur.get('temperature_2m', '—')} °C",
            ]
        )
    return "\n".join(lines)


async def _resolve_location(
    city: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    coordinates: Optional[str] = None,
) -> Tuple[float, float, str, str]:
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude), f"{latitude},{longitude}", "auto"
    if coordinates:
        parsed = _parse_lat_lon(coordinates)
        if parsed:
            lat, lon = parsed
            return lat, lon, f"{lat},{lon}", "auto"
    if city:
        geo = await _geocode_city(city)
        return (
            geo["latitude"],
            geo["longitude"],
            _place_label(geo),
            geo["timezone"],
        )
    raise ValueError("请提供 city（城市名）或 latitude+longitude / coordinates（纬度,经度）")


def register_weather_tools(mcp) -> None:
    """向 FastMCP 实例注册天气相关工具。"""

    @mcp.tool()
    async def weather_current(
        city: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        coordinates: str = "",
    ) -> str:
        """
        查询指定地点的当前天气（实况）。
        city：城市中文或英文名，如「广州」「北京」；与经纬度二选一。
        latitude/longitude：十进制度坐标；或 coordinates 形如「23.13,113.26」。
        数据来源：Open-Meteo（免费，无需 API Key）。
        """
        try:
            lat, lon, place, tz = await _resolve_location(
                city=city or None,
                latitude=latitude,
                longitude=longitude,
                coordinates=coordinates or None,
            )
            data = await _fetch_forecast(lat, lon, timezone=tz, forecast_days=1)
            return _format_current(data, place)
        except httpx.HTTPError as e:
            return f"天气服务请求失败：{e}"
        except Exception as e:
            return f"查询失败：{e}"

    @mcp.tool()
    async def weather_forecast(
        city: str = "",
        days: int = 3,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        coordinates: str = "",
    ) -> str:
        """
        查询指定地点的多日天气预报（含每日最高/最低温与降水）。
        city：城市名；days：预报天数 1~7（默认 3）。
        也可使用 latitude/longitude 或 coordinates 指定坐标。
        """
        try:
            lat, lon, place, tz = await _resolve_location(
                city=city or None,
                latitude=latitude,
                longitude=longitude,
                coordinates=coordinates or None,
            )
            n = max(1, min(7, int(days)))
            data = await _fetch_forecast(lat, lon, timezone=tz, forecast_days=n)
            return _format_forecast(data, place, n)
        except httpx.HTTPError as e:
            return f"天气服务请求失败：{e}"
        except Exception as e:
            return f"查询失败：{e}"

    @mcp.tool()
    async def weather_search_city(keyword: str, limit: int = 5) -> str:
        """
        按关键字搜索城市/地点（地理编码），返回候选列表及经纬度，便于后续调用 weather_current。
        keyword：地名关键字；limit：最多返回条数（默认 5，最大 10）。
        """
        kw = (keyword or "").strip()
        if not kw:
            return "请提供 keyword，例如：东莞、杭州"
        lim = max(1, min(10, int(limit)))
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": kw,
                        "count": lim,
                        "language": "zh",
                        "format": "json",
                    },
                )
                r.raise_for_status()
                data = r.json()
            results = data.get("results") or []
            if not results:
                return f"未找到与「{kw}」相关的地点。"
            lines = [f"共 {len(results)} 条候选：", ""]
            for i, hit in enumerate(results, 1):
                label = " · ".join(
                    p
                    for p in [
                        hit.get("name"),
                        hit.get("admin1"),
                        hit.get("country"),
                    ]
                    if p
                )
                lines.append(
                    f"{i}. {label}\n"
                    f"   纬度 {hit.get('latitude')}，经度 {hit.get('longitude')}\n"
                    f"   时区 {hit.get('timezone', 'auto')}"
                )
            return "\n".join(lines)
        except httpx.HTTPError as e:
            return f"地理编码请求失败：{e}"
        except Exception as e:
            return f"搜索失败：{e}"

    # 可选：和风天气（需 .env 中 QWEATHER_API_KEY）
    qkey = os.getenv("QWEATHER_API_KEY", "").strip()

    if qkey:

        @mcp.tool()
        async def weather_qweather_now(city: str = "", location_id: str = "") -> str:
            """
            使用和风天气查询实时天气（需在环境变量配置 QWEATHER_API_KEY）。
            city：城市名（将先查 location_id）；或直接传 location_id（和风 LocationID）。
            """
            try:
                loc_id = (location_id or "").strip()
                if not loc_id and city:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        gr = await client.get(
                            "https://geoapi.qweather.com/v2/city/lookup",
                            params={
                                "location": city.strip(),
                                "key": qkey,
                                "number": 1,
                                "lang": "zh",
                            },
                        )
                        gr.raise_for_status()
                        gdata = gr.json()
                    locs = (gdata.get("location") or []) if gdata.get("code") == "200" else []
                    if not locs:
                        return f"和风未找到城市：{city}"
                    loc = locs[0]
                    loc_id = loc["id"]
                    place = f"{loc.get('name')} ({loc.get('adm1')}, {loc.get('country')})"
                elif loc_id:
                    place = f"LocationID={loc_id}"
                else:
                    return "请提供 city 或 location_id"

                async with httpx.AsyncClient(timeout=20.0) as client:
                    wr = await client.get(
                        "https://devapi.qweather.com/v7/weather/now",
                        params={"location": loc_id, "key": qkey, "lang": "zh"},
                    )
                    wr.raise_for_status()
                    wdata = wr.json()
                if wdata.get("code") != "200":
                    return f"和风天气错误：{wdata.get('code')} {wdata.get('msg', '')}"
                now = wdata.get("now") or {}
                lines = [
                    f"地点：{place}",
                    f"观测：{wdata.get('updateTime', '—')}",
                    f"天气：{now.get('text', '—')}",
                    f"气温：{now.get('temp', '—')} °C，体感 {now.get('feelsLike', '—')} °C",
                    f"湿度：{now.get('humidity', '—')} %",
                    f"风向：{now.get('windDir', '—')} {now.get('windScale', '—')}级",
                    f"降水：{now.get('precip', '—')} mm",
                ]
                return "\n".join(lines)
            except httpx.HTTPError as e:
                return f"和风 API 请求失败：{e}"
            except Exception as e:
                return f"查询失败：{e}"
