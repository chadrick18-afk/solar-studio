"""PV design + hourly simulation API with market-driven BOM costing.
Run: uvicorn pv_api:app --reload
Deps: pip install fastapi uvicorn pvlib pandas
"""
from __future__ import annotations

import calendar
import time
from functools import lru_cache
from math import ceil, pi, sin, sqrt
from pathlib import Path
from typing import Literal, Optional

import pvlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="PV Simulation API")

TARIFFS_AS_OF = "Q1 2026 - Meralco / Regional Utility Baseline"
DU_RATES = {
    "meralco": {"retail": 14.50, "blended_export": 6.00},
    "veco": {"retail": 12.80, "blended_export": 5.50},
    "dlpc": {"retail": 11.50, "blended_export": 5.20},
    "generic_coop": {"retail": 13.00, "blended_export": 5.80},
}
DAYS = [calendar.monthrange(2023, m)[1] for m in range(1, 13)]
MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]

# Real-world PH retail market unit pricing parameters (PHP)
UNIT_PRICES = {
    "panel_per_watt": 7.5,           # ₱7.5/W for Tier-1 Mono modules
    "inverter_per_kw": 8500,         # ₱8,500/kW for Hybrid/Grid-tied inverters
    "bat_kwh": 8500,                 # ₱8,500/kWh for 51.2V LiFePO4 modules
    "rail_4_2m": 650,                # ₱650 per 4.2m aluminum solar mounting rail
    "roof_hook": 120,                # ₱120 per stainless roof hook/foot
    "clamp": 45,                     # ₱45 per mid/end clamp
    "pv_cable_per_m": 45,            # ₱45/m for twin 4-6mm² DC solar cable
    "ac_cable_per_m": 85,            # ₱85/m for THHN AC feeder wire
    "bat_cable_per_m": 350,          # ₱350/m for 25-50mm² flexible battery cable
    # Protection (itemized, PHP each)
    "prot_pv_dc_breaker": 1800,      # PV DC breaker / isolator
    "prot_ac_breaker": 900,          # AC breaker
    "prot_spd_dc": 1300,             # DC surge protective device
    "prot_spd_ac": 1100,             # AC surge protective device
    "prot_battery_mccb": 4500,       # Battery DC MCCB (only when added in the wiring editor)
    "prot_enclosure_misc": 1400,     # Combiner / MDB enclosure, busbars, DIN rail
    # Conduit & grounding (itemized, PHP lot each)
    "conduit_fittings": 2200,        # Conduit pipe, elbows, couplings, clips
    "grounding_kit": 1500,           # Ground rod, clamp, bare copper
    "lugs_glands_ties": 1000,        # Lugs, glands, cable ties, labels, MC4 pairs

    # Separated Services & Soft Costs
    "standard_labor_per_kw": 3500,   # Mechanical & electrical installation labor rate per kWp
    "pee_sign_and_seal": 15000,      # Professional Electrical Engineer (PEE) Sign & Seal
    "net_metering_fee": 12000,       # Utility DIS (Distribution Impact Study), meter testing, & filing
}


def _norm(w):
    t = sum(w)
    return [x / t for x in w]


PROFILES = {
    "aircon_day": _norm([2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 7, 6, 6, 6, 5, 4, 3, 2]),
    "evening_peak": _norm([3, 3, 3, 3, 3, 3, 4, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 8, 9, 9, 8, 6, 4]),
    "flat": _norm([1] * 24),
}
SUN = range(7, 18)

# Roof profiles sent by the roof mapper. "loss" is a yield allowance applied to that face's PVGIS output.
# flat: panels sit on tilted racks, so rows need spacing; 3% is a rough allowance for row-to-row shading.
# Other profiles get no extra loss. These are planning assumptions, not measured values.
ROOF_TYPES = {"custom": 0.0, "flat": 0.03, "shed": 0.0, "gable": 0.0, "hip": 0.0, "metal": 0.0, "tile": 0.0}
RoofType = Literal["custom", "flat", "shed", "gable", "hip", "metal", "tile"]


def _derate(y, loss):
    """Return a copy of an hourly_yield() result with the output scaled by (1 - loss)."""
    if not loss:
        return y
    k = 1 - loss
    return [v * k for v in y[0]], y[1], y[2]


class Appliance(BaseModel):
    name: str = Field("", max_length=60)
    watts: float = Field(gt=0, le=20000, description="running watts per unit")
    qty: int = Field(1, ge=1, le=50)
    hours: float = Field(gt=0, le=24, description="hours per day")
    start: int = Field(0, ge=0, le=23, description="hour of day it starts")
    surge: float = Field(1, ge=1, le=10, description="start-up current multiple")


def appliance_profile(apps, daily):
    e, pk, extra = [0.0] * 24, [0.0] * 24, 0.0
    for a in apps:
        kw = a.watts * a.qty / 1000
        extra = max(extra, a.watts * (a.surge - 1) / 1000)
        for h in range(ceil(a.hours)):
            i = (a.start + h) % 24
            e[i] += kw * min(1.0, a.hours - h)
            pk[i] += kw
    tot = sum(e)
    base = max(0.0, daily - tot)
    prof = _norm([x + base / 24 for x in e])
    return prof, max(x + base / 24 for x in pk), extra, tot, base


class Face(BaseModel):
    name: str = Field("", max_length=40)
    tilt: float = Field(15, ge=0, le=90)
    azimuth: float = Field(180, ge=0, le=360)
    max_panels: Optional[int] = Field(None, ge=1, le=2000, description="roof capacity from the mapper")
    roof_type: RoofType = Field("custom", description="roof profile chosen in the roof mapper")


class SimRequest(BaseModel):
    topology: Literal["grid-tied", "hybrid-no-export", "hybrid-export", "off-grid"] = "hybrid-no-export"
    du_provider: Literal["meralco", "veco", "dlpc", "generic_coop"] = "meralco"
    load_profile: Literal["aircon_day", "evening_peak", "flat"] = "aircon_day"
    appliances: list[Appliance] = Field(default_factory=list, max_length=30)
    monthly_bill: Optional[float] = Field(None, gt=0)
    monthly_kwh: Optional[float] = Field(None, gt=0)
    lat: float = Field(14.5995, ge=-90, le=90)
    lon: float = Field(120.9842, ge=-180, le=180)
    tilt: float = Field(15, ge=0, le=90)
    azimuth: float = Field(180, ge=0, le=360)
    target_offset: float = Field(0.8, gt=0, le=1.2)
    system_kw: Optional[float] = Field(None, gt=0)
    inverter_kw: Optional[float] = Field(None, gt=0, le=200)
    panel_watts: int = Field(450, ge=100, le=800)
    battery_kwh: Optional[float] = Field(None, ge=0)
    autonomy_days: float = Field(1, ge=0.5, le=5)
    battery_roundtrip_eff: float = Field(0.90, gt=0, le=1)
    battery_cutoff_pct: float = Field(20, ge=0, lt=80)
    export_priority: Literal["battery_first", "export_first"] = Field(
        "battery_first", description="surplus PV order when export is allowed: charge battery first, or export first")
    retail_rate: Optional[float] = Field(None, gt=0)
    export_rate: Optional[float] = Field(None, ge=0)
    cost_per_kw_php: float = Field(65_000, gt=0)
    battery_cost_per_kwh_php: float = Field(22_000, ge=0)
    degradation_pct: float = Field(0.5, ge=0, le=5)
    tariff_escalation_pct: float = Field(3.0, ge=0, le=15)
    inverter_replacement_pct: float = Field(8.0, ge=0, le=50)
    years: int = Field(25, ge=1, le=40)
    pv_run_m: float = Field(25, ge=0, le=500, description="array to inverter cable run (m)")
    ac_run_m: float = Field(8, ge=0, le=500, description="inverter to breaker panel AC run (m)")
    bat_run_m: float = Field(3, ge=0, le=100, description="battery to inverter cable run (m)")
    extra_materials_php: float = Field(0, ge=0, le=10_000_000, description="materials added in the wiring editor beyond the standard set")
    array_rows: Optional[int] = Field(None, ge=1, le=500, description="panel rows in the layout (for clamp counts)")
    faces: list[Face] = Field(default_factory=list, max_length=6, description="roof faces, each simulated with its own tilt/azimuth")
    price_overrides: dict[str, float] = Field(default_factory=dict, description="unit price overrides keyed like UNIT_PRICES; unknown keys are ignored")

    @model_validator(mode="after")
    def need_consumption(self):
        if self.monthly_bill is None and self.monthly_kwh is None:
            raise ValueError("Provide monthly_kwh or monthly_bill")
        if len(self.price_overrides) > 60:
            raise ValueError("Too many price overrides")
        return self


@lru_cache(maxsize=64)
def _pvgis_hourly(lat, lon, tilt, azimuth) -> tuple:
    import pandas as pd
    res = pvlib.iotools.get_pvgis_hourly(
        lat, lon, start=2015, end=2020, raddatabase="PVGIS-ERA5", pvcalculation=True,
        peakpower=1, loss=14, surface_tilt=tilt, surface_azimuth=azimuth, timeout=30)
    p = res[0]["P"] / 1000
    dly = p.resample("D").sum()
    mean_m = dly.groupby(dly.index.month).transform("mean")
    worst = float((dly.rolling(3).sum() / mean_m.rolling(3).sum()).min())
    if not (worst == worst):
        worst = 0.3
    worst = min(1.0, max(0.05, worst))
    idx = p.index.tz_localize(None) + pd.Timedelta(hours=round(lon / 15))
    g = p.groupby([idx.month, idx.day, idx.hour]).mean()
    lv = g.index.get_level_values
    g = g[~((lv(0) == 2) & (lv(1) == 29))]
    if len(g) != 8760:
        raise ValueError("unexpected PVGIS length")
    return tuple(float(x) for x in g.tolist()), worst


_FAIL: dict = {}


def hourly_yield(lat, lon, tilt, azimuth):
    key = (round(lat, 2), round(lon, 2), round(tilt), round(azimuth))
    if time.monotonic() - _FAIL.get(key, -1e9) > 300:
        try:
            vals, worst = _pvgis_hourly(*key)
            return list(vals), "PVGIS-ERA5 hourly (2015-2020 typical year)", worst
        except Exception:
            _FAIL[key] = time.monotonic()
    shape = [max(0.0, sin(pi * (h - 6) / 12)) for h in range(24)]
    k = 4.5 * 0.86 / sum(shape)
    return [s * k for _ in range(365) for s in shape], "FALLBACK synthetic 4.5 PSH day (PVGIS unavailable)", 0.3


def dispatch(pv, kw, monthly_kwh, prof, bat_kwh, rt, usable, power_kw, grid, cut=0.0, export_first=False):
    e, cap = sqrt(rt), bat_kwh * usable
    soc = cap * 0.5
    daily = monthly_kwh * 12 / 365
    keys = ("gen", "direct", "discharge", "export", "curtail", "grid_import", "unserved")
    mon = [dict.fromkeys(keys, 0.0) for _ in range(12)]
    day = [{k: [0.0] * 24 for k in ("pv", "load", "soc", "grid")} for _ in range(12)]
    i = 0
    for m, days in enumerate(DAYS):
        for _ in range(days):
            for h in range(24):
                g, l = pv[i] * kw, daily * prof[h]
                i += 1
                r = mon[m]
                r["gen"] += g
                d = min(g, l)
                g, l = g - d, l - d
                r["direct"] += d
                dd, dis = day[m], 0.0
                dd["pv"][h] += (g + d)
                dd["load"][h] += (l + d)
                if cap > 0:
                    if g > 0 and not export_first:   # export-first: surplus goes to the grid, battery only discharges
                        c = min(g, power_kw, (cap - soc) / e)
                        soc += c * e
                        g -= c
                    if l > 0:
                        dis = min(l, power_kw, soc * e)
                        soc -= dis / e
                        l -= dis
                r["discharge"] += dis
                r["export"] += g if grid == "export" else 0.0
                r["curtail"] += g if grid != "export" else 0.0
                if grid == "offgrid":
                    r["unserved"] += l
                else:
                    r["grid_import"] += l
                    dd["grid"][h] += l
                dd["soc"][h] += (cut + soc / cap * (100 - cut)) if cap else 0.0
        for k in ("pv", "load", "soc", "grid"):
            day[m][k] = [round(v / days, 3) for v in day[m][k]]
    return mon, day


def stress_test(pv, kw, daily, prof, bat_kwh, rt, usable, power_kw, worst, days=3):
    sums, i = [], 0
    for n in DAYS:
        sums.append(sum(pv[i:i + n * 24]))
        i += n * 24
    m = min(range(12), key=lambda k: sums[k] / DAYS[k])
    start = sum(DAYS[:m]) * 24
    mean_day = [sum(pv[start + d * 24 + h] for d in range(DAYS[m])) / DAYS[m] for h in range(24)]
    e, cap = sqrt(rt), bat_kwh * usable
    soc, unserved = cap, 0.0
    for _ in range(days):
        for h in range(24):
            g, l = mean_day[h] * kw * worst, daily * prof[h]
            d = min(g, l)
            g, l = g - d, l - d
            if cap > 0:
                if g > 0:
                    soc += min(g, power_kw, (cap - soc) / e) * e
                if l > 0:
                    dis = min(l, power_kw, soc * e)
                    soc -= dis / e
                    l -= dis
            unserved += l
    return unserved, daily * days, MONTHS[m]


def cash_flow(y1, pv_cost, total, req):
    deg, esc, cum, pay, out = req.degradation_pct / 100, req.tariff_escalation_pct / 100, -total, None, []
    for y in range(1, req.years + 1):
        gross = y1 * (1 - deg) ** (y - 1) * (1 + esc) ** (y - 1)
        capex = pv_cost * req.inverter_replacement_pct / 100 if y == 12 else 0.0
        net, prev = gross - capex, cum
        cum += net
        if pay is None and cum >= 0:
            pay = (y - 1) + (-prev / net if net > 0 else 1)
        out.append({"year": y, "savings_php": round(gross), "replacement_php": round(capex), "cumulative_php": round(cum)})
    return out, pay


@app.post("/api/simulate")
def simulate(req: SimRequest):
    rates = DU_RATES[req.du_provider]
    retail = req.retail_rate or rates["retail"]
    exp_rate = req.export_rate if req.export_rate is not None else rates["blended_export"]
    usable = 1 - req.battery_cutoff_pct / 100
    grid = {"grid-tied": "export", "hybrid-export": "export", "hybrid-no-export": "zero", "off-grid": "offgrid"}[req.topology]
    notes = [f"Tariffs as of: {TARIFFS_AS_OF}"]

    monthly_kwh = req.monthly_kwh
    if monthly_kwh is None:
        monthly_kwh = req.monthly_bill / retail
        notes.append("kWh estimated as bill / retail rate; prefer monthly_kwh")
    annual = monthly_kwh * 12
    daily = annual / 365
    peak_run = surge_extra = None
    if req.appliances:
        prof, peak_run, surge_extra, app_kwh, base = appliance_profile(req.appliances, daily)
        notes.append(f"Appliance list covers {app_kwh:.1f} kWh/day ({app_kwh / daily * 100:.0f}% of your {daily:.1f} kWh/day); the rest is a flat baseline of {base:.1f} kWh/day")
    else:
        prof = PROFILES[req.load_profile]
    day_frac = sum(prof[h] for h in SUN)

    if req.faces:
        fy = [_derate(hourly_yield(req.lat, req.lon, f.tilt, f.azimuth), ROOF_TYPES[f.roof_type]) for f in req.faces]
        for f in req.faces:
            if ROOF_TYPES[f.roof_type]:
                notes.append(f"{f.name or f.roof_type}: {ROOF_TYPES[f.roof_type] * 100:.0f}% yield allowance for {f.roof_type}-roof row spacing and shading; leave extra room between rows.")
            if f.azimuth < 90 or f.azimuth > 270:
                notes.append(f"{f.name or f.roof_type} faces away from the sun (azimuth {f.azimuth:.0f}°), so its yield is much lower.")
        specs = [sum(y[0]) for y in fy]
        src = next((y[1] for y in fy if y[1].startswith("FALLBACK")), fy[0][1])
        pv, worst = fy[0][0], fy[0][2]          # placeholder until panels are allocated
    else:
        pv, src, worst = hourly_yield(req.lat, req.lon, req.tilt, req.azimuth)
    spec_yield = sum(pv)

    has_bat = req.topology != "grid-tied"
    if not has_bat:
        bat = 0.0
    elif req.battery_kwh is not None:
        bat = req.battery_kwh
    elif req.topology == "off-grid":
        bat = daily * req.autonomy_days / usable
    else:
        bat = daily * (1 - day_frac) / (usable * req.battery_roundtrip_eff)
    bat = (round(bat * 2) / 2 if req.battery_kwh is None else round(bat, 3)) if bat else 0.0

    pw = req.panel_watts / 1000
    if req.faces:
        order = sorted(range(len(req.faces)), key=lambda j: -specs[j])   # best face first
        counts = [0] * len(req.faces)
        if req.system_kw is None:
            left = annual * (1.0 if bat > 0 else day_frac) * req.target_offset   # kWh/yr to cover
            for j in order:
                cap, per = req.faces[j].max_panels, pw * specs[j]            # per = kWh/yr per panel
                n = ceil(left / per - 1e-9) if left > 0 else 0
                counts[j] = min(n, cap) if cap else n
                left -= counts[j] * per
            if left > 1e-6:
                notes.append("Roof faces are too small to reach the target offset; all faces are full.")
            if sum(counts) == 0:
                counts[order[0]] = 1
        else:
            left = max(1, ceil(round(req.system_kw * 1000 / req.panel_watts, 6)))
            for j in order:
                n = min(left, req.faces[j].max_panels or left)
                counts[j], left = n, left - n
            if left:
                counts[order[0]] += left
                notes.append(f"{left} panel(s) exceed the roof capacity; placed on the best face.")
        panels = sum(counts)
    else:
        if req.system_kw is None:
            basis = annual * (1.0 if bat > 0 else day_frac)
            raw = basis * req.target_offset / spec_yield
        else:
            raw = req.system_kw
        panels = max(1, ceil(round(raw * 1000 / req.panel_watts, 6)))
    kw = panels * req.panel_watts / 1000

    face_rows = []
    if req.faces:
        kws = [c * pw for c in counts]
        pv = [sum(kws[j] * fy[j][0][i] for j in range(len(fy))) / kw for i in range(8760)]
        worst = sum(kws[j] * fy[j][2] for j in range(len(fy))) / kw
        spec_yield = sum(pv)
        face_rows = [{"name": f.name or f"Face {j+1}", "panels": counts[j], "kwp": round(kws[j], 2),
                      "tilt": f.tilt, "azimuth": f.azimuth, "roof_type": f.roof_type,
                      "specific_yield_kwh_per_kwp": round(specs[j]),
                      "annual_kwh": round(kws[j] * specs[j])} for j, f in enumerate(req.faces)]
        notes.append(f"Yield modeled per roof face ({len(req.faces)} faces) and combined by kWp.")

    if peak_run is not None:
        inv_kw = max(1, ceil(max((peak_run + surge_extra) * 1.1, kw / 1.25)))
        notes.append(f"Inverter sized for peak running load {peak_run:.1f} kW + largest start surge {surge_extra:.1f} kW (+10% margin)")
    else:
        inv_kw = max(1, ceil(max(daily * max(prof) * 1.5, kw / 1.25)))
    if req.inverter_kw:
        inv_kw = round(req.inverter_kw, 2)
        notes.append(f"Inverter size set by method/override: {inv_kw} kW")

    export_first = grid == "export" and bat > 0 and req.export_priority == "export_first"
    if export_first:
        notes.append("Export-first dispatch: surplus PV is exported before charging the battery, so the battery stays at its starting charge and adds little.")
    mon, day = dispatch(pv, kw, monthly_kwh, prof, bat, req.battery_roundtrip_eff,
                        usable, inv_kw, grid, req.battery_cutoff_pct, export_first)
    stress_pct = stress_kwh = None
    if req.topology == "off-grid":
        un, tot, wm = stress_test(pv, kw, daily, prof, bat, req.battery_roundtrip_eff, usable, inv_kw, worst)
        stress_kwh, stress_pct = round(un, 1), round(100 * un / tot, 1) if tot else 0.0

    rows = []
    for m, r in enumerate(mon):
        sav = (r["direct"] + r["discharge"]) * retail + r["export"] * exp_rate
        rows.append({"month": MONTHS[m], **{k: round(v, 1) for k, v in r.items()}, "savings_php": round(sav, 2)})
    y1 = sum(r["savings_php"] for r in rows)
    gen = sum(r["gen"] for r in rows)
    onsite = sum(r["direct"] + r["discharge"] for r in rows)

    # Itemized materials (BOM). Labor, engineering and permitting are kept OUT of the BOM.
    U = {**UNIT_PRICES, **{k: float(v) for k, v in req.price_overrides.items() if k in UNIT_PRICES and 0 <= v <= 10_000_000}}
    cost_panels = kw * 1000 * U["panel_per_watt"]
    cost_inverter = inv_kw * U["inverter_per_kw"]
    cost_battery = bat * U["bat_kwh"]

    rows_ = req.array_rows or ceil(panels / 12)
    rails_qty = ceil(panels * 2.26 / 4.2)
    hooks_qty = ceil(panels * 1.5)
    mid_qty = max(0, panels - rows_) * 2
    end_qty = rows_ * 4
    cost_mounting = (rails_qty * U["rail_4_2m"] + hooks_qty * U["roof_hook"]
                     + (mid_qty + end_qty) * U["clamp"])

    cost_protection = (U["prot_pv_dc_breaker"] + U["prot_ac_breaker"] + U["prot_spd_dc"]
                       + U["prot_spd_ac"] + U["prot_enclosure_misc"])
    cost_conduit_ground = U["conduit_fittings"] + U["grounding_kit"] + U["lugs_glands_ties"]
    cost_cables = (req.pv_run_m * 2 * U["pv_cable_per_m"] + req.ac_run_m * 3 * U["ac_cable_per_m"]
                   + (req.bat_run_m * 2 * U["bat_cable_per_m"] if bat > 0 else 0.0))
    cost_electrical = cost_protection + cost_conduit_ground + cost_cables

    cost_extra = req.extra_materials_php
    bom_materials = cost_panels + cost_inverter + cost_battery + cost_mounting + cost_electrical + cost_extra

    # Services & fees (NOT in the BOM, but still part of project cost / payback)
    cost_labor = kw * U["standard_labor_per_kw"]
    cost_pee = U["pee_sign_and_seal"]
    cost_net_metering = U["net_metering_fee"] if req.topology in ["grid-tied", "hybrid-export"] else 0.0
    services = cost_labor + cost_pee + cost_net_metering

    # Financial aggregations
    pv_cost = bom_materials - cost_battery + services
    total = pv_cost + cost_battery
    flows, pay = cash_flow(y1, pv_cost, total, req)

    return {
        "summary": {
            "topology": req.topology, "faces": face_rows, "system_kw": round(kw, 2), "panel_count": panels, "panel_watts": req.panel_watts,
            "battery_kwh": bat, "battery_cutoff_pct": req.battery_cutoff_pct, "peak_running_kw": round(peak_run, 2) if peak_run is not None else None, "inverter_kw": inv_kw, "specific_yield_kwh_per_kwp": round(spec_yield),
            "annual_generation_kwh": round(gen), "solar_coverage_pct": round(100 * onsite / annual, 1),
            "self_consumption_pct": round(100 * onsite / gen, 1) if gen else 0,
            "unserved_kwh": round(sum(r["unserved"] for r in rows)),
            "stress_unserved_kwh": stress_kwh, "stress_unserved_pct": stress_pct,
            "year1_savings_php": round(y1), "pv_cost_php": round(pv_cost),
            "battery_cost_php": round(cost_battery), "bom_materials_php": round(bom_materials), "extra_materials_php": round(cost_extra),
            "services_php": round(services), "total_cost_php": round(total),
            "payback_years": round(pay, 1) if pay is not None else None,
            "net_gain_over_horizon_php": flows[-1]["cumulative_php"],
        },
        "monthly": rows, "typical_day": day, "cash_flow": flows,
        "assumptions": {"monthly_kwh": round(monthly_kwh, 1), "retail_rate": retail, "export_rate": exp_rate,
                        "yield_source": src, "load_profile": "appliance list" if req.appliances else req.load_profile,
                        "notes": notes, "unit_prices": U},
    }


_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")