"""Measure endpoint latency over N Azure ticks."""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from dataclasses import dataclass, field

import requests

from hardware.cloud import fetch_cloud_snapshot
from hardware.config import (
    CLOUD_BASE_URL,
    FLASK_PORT,
    LAPTOP_IP,
    PICO_CAP_DATA_URL,
    PICO_ENDPOINTS,
    TICK_INTERVAL_S,
)
from hardware.poller import read_hardware_for_inference

TICK_POLL_S = 0.25
TIMEOUT_S = 1.0

AZURE_PATHS = ("price", "demand", "deferables")
PICO_LABELS = ("pvout", "vcap")


@dataclass
class EndpointResult:
    ok: bool
    ms: float
    detail: str = ""


@dataclass
class TickReport:
    index: int
    expected_tick: int | None
    actual_tick: int | None
    day: int | None
    skipped: int
    wait_ms: float
    azure: dict[str, EndpointResult] = field(default_factory=dict)
    pico: dict[str, EndpointResult] = field(default_factory=dict)
    azure_parallel_ms: float = 0.0
    pico_parallel_ms: float = 0.0
    inference_fetch_ms: float = 0.0
    flask_ms: float | None = None
    flask_ok: bool | None = None
    cycle_ms: float = 0.0
    over_budget: bool = False


def _timed_get(url: str, *, timeout_s: float = TIMEOUT_S) -> EndpointResult:
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=timeout_s)
        ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        body = r.text.strip().replace("\n", " ")[:60]
        return EndpointResult(True, ms, body)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return EndpointResult(False, ms, f"{type(e).__name__}: {e}")


def _fetch_azure_price() -> tuple[EndpointResult, dict | None]:
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{CLOUD_BASE_URL}/price", timeout=TIMEOUT_S)
        ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        payload = r.json()
        detail = f"tick={payload.get('tick')} day={payload.get('day')}"
        return EndpointResult(True, ms, detail), payload
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return EndpointResult(False, ms, f"{type(e).__name__}: {e}"), None


def _fetch_azure_sequential() -> dict[str, EndpointResult]:
    out: dict[str, EndpointResult] = {}
    for name in AZURE_PATHS:
        out[name] = _timed_get(f"{CLOUD_BASE_URL}/{name}")
    return out


def _fetch_azure_parallel() -> tuple[dict[str, EndpointResult], float]:
    t0 = time.perf_counter()

    def one(name: str) -> tuple[str, EndpointResult]:
        return name, _timed_get(f"{CLOUD_BASE_URL}/{name}")

    out: dict[str, EndpointResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for name, res in pool.map(one, AZURE_PATHS):
            out[name] = res
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return out, wall_ms


def _fetch_pico_sequential() -> dict[str, EndpointResult]:
    out: dict[str, EndpointResult] = {}
    pv = PICO_ENDPOINTS["pvout"]
    out["pvout"] = _timed_get(f"{pv['base_url']}{pv['path']}")
    out["vcap"] = _timed_get(f"{PICO_CAP_DATA_URL}/data")
    return out


def _fetch_pico_parallel() -> tuple[dict[str, float | None], float]:
    t0 = time.perf_counter()
    readings = read_hardware_for_inference()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return readings, wall_ms


def _wait_for_tick(last_tick: int | None, last_day: int | None) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    while True:
        res, payload = _fetch_azure_price()
        if not res.ok or payload is None:
            time.sleep(TICK_POLL_S)
            continue
        tick = payload.get("tick")
        day = payload.get("day")
        if tick is None or day is None:
            time.sleep(TICK_POLL_S)
            continue
        tick_i, day_i = int(tick), int(day)
        if last_tick is None:
            return tick_i, day_i, (time.perf_counter() - t0) * 1000.0
        if day_i != last_day or tick_i > last_tick:
            return tick_i, day_i, (time.perf_counter() - t0) * 1000.0
        time.sleep(TICK_POLL_S)


def _probe_flask() -> EndpointResult:
    for host in (LAPTOP_IP, "127.0.0.1"):
        res = _timed_get(f"http://{host}:{FLASK_PORT}/api/state", timeout_s=2.0)
        if res.ok:
            res.detail = f"host={host} {res.detail}"
            return res
    return res


def run_benchmark(n_ticks: int) -> list[TickReport]:
    reports: list[TickReport] = []
    last_tick: int | None = None
    last_day: int | None = None

    print("Inference I/O path (no sun, no cap /p):")
    print(f"  Azure: {', '.join(AZURE_PATHS)}")
    print(f"  Pico PV: {PICO_ENDPOINTS['pvout']['base_url']}{PICO_ENDPOINTS['pvout']['path']}")
    print(f"  Pico vcap: {PICO_CAP_DATA_URL}/data")
    print(f"  Flask: http://{LAPTOP_IP}:{FLASK_PORT}/api/state")
    print(f"  Budget per tick: {TICK_INTERVAL_S * 1000:.0f} ms\n")

    flask_probe = _probe_flask()
    print(
        f"Flask probe: {'OK' if flask_probe.ok else 'FAIL'} "
        f"({flask_probe.ms:.0f} ms) {flask_probe.detail}\n"
    )

    for i in range(n_ticks):
        cycle_t0 = time.perf_counter()
        expected = (last_tick + 1) if last_tick is not None else None

        tick_seen, day_seen, wait_ms = _wait_for_tick(last_tick, last_day)
        skipped = 0
        if last_tick is not None and day_seen == last_day and tick_seen > last_tick + 1:
            skipped = tick_seen - last_tick - 1

        inf_t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            cloud_f = pool.submit(fetch_cloud_snapshot)
            hw_f = pool.submit(read_hardware_for_inference)
            cloud_snap = cloud_f.result()
            pico_readings = hw_f.result()
        inference_fetch_ms = (time.perf_counter() - inf_t0) * 1000.0

        azure_seq = _fetch_azure_sequential()
        pico_seq = _fetch_pico_sequential()

        flask_ms = None
        flask_ok = None
        if flask_probe.ok:
            fr = _probe_flask()
            flask_ms = fr.ms
            flask_ok = fr.ok

        cycle_ms = wait_ms + inference_fetch_ms
        report = TickReport(
            index=i + 1,
            expected_tick=expected,
            actual_tick=tick_seen,
            day=day_seen,
            skipped=skipped,
            wait_ms=wait_ms,
            azure=azure_seq,
            pico=pico_seq,
            azure_parallel_ms=inference_fetch_ms,
            pico_parallel_ms=inference_fetch_ms,
            inference_fetch_ms=inference_fetch_ms,
            flask_ms=flask_ms,
            flask_ok=flask_ok,
            cycle_ms=cycle_ms,
            over_budget=cycle_ms > TICK_INTERVAL_S * 1000.0,
        )
        reports.append(report)

        print(f"=== Tick cycle {i + 1}/{n_ticks} ===")
        print(
            f"  Azure tick: {tick_seen} day={day_seen} "
            f"(expected={expected}, skipped={skipped}, wait={wait_ms:.0f}ms)"
        )
        for name in AZURE_PATHS:
            res = azure_seq[name]
            status = "OK" if res.ok else "FAIL"
            print(f"  azure/{name:11s} {status:4s} {res.ms:7.1f}ms  {res.detail}")
        azure_seq_total = sum(r.ms for r in azure_seq.values())
        snap_tick = cloud_snap.get("tick")
        print(
            f"  azure sequential(diag)={azure_seq_total:.0f}ms  "
            f"inference parallel fetch={inference_fetch_ms:.0f}ms  "
            f"snapshot tick={snap_tick}"
        )

        for name in PICO_LABELS:
            res = pico_seq[name]
            status = "OK" if res.ok else "FAIL"
            print(f"  pico/{name:12s} {status:4s} {res.ms:7.1f}ms  {res.detail}")
        pico_seq_total = sum(r.ms for r in pico_seq.values())
        pv_v = pico_readings.get("pvout")
        vc_v = pico_readings.get("vcap")
        print(
            f"  pico sequential(diag)={pico_seq_total:.0f}ms  "
            f"values pv={pv_v} vcap={vc_v}"
        )

        if flask_ms is not None:
            print(f"  flask/state   {'OK' if flask_ok else 'FAIL':4s} {flask_ms:7.1f}ms")
        print(
            f"  BUDGET (wait+fetch): {cycle_ms:.0f}ms / {TICK_INTERVAL_S * 1000:.0f}ms "
            f"{'OVER BUDGET' if report.over_budget else 'OK'}"
        )
        print()

        last_tick = tick_seen
        last_day = day_seen

    return reports


def _summarize(reports: list[TickReport]) -> None:
    n = len(reports)
    skipped_ticks = sum(r.skipped for r in reports)
    over_budget = sum(1 for r in reports if r.over_budget)
    failures: dict[str, int] = {}

    def bump(key: str) -> None:
        failures[key] = failures.get(key, 0) + 1

    for r in reports:
        if r.skipped:
            bump("tick_skip")
        if r.over_budget:
            bump("over_budget")
        for name, res in {**r.azure, **r.pico}.items():
            if not res.ok:
                bump(name)

    print("=" * 60)
    print("SUMMARY (inference path: no sun, no cap /p)")
    print(f"  Cycles measured:              {n}")
    print(f"  Azure ticks skipped:          {skipped_ticks}")
    print(f"  Cycles over 5s:               {over_budget}/{n}")
    if reports:
        mean_inf = sum(r.inference_fetch_ms for r in reports) / n
        mean_cycle = sum(r.cycle_ms for r in reports) / n
        print(f"  Mean inference fetch:         {mean_inf:.0f}ms")
        print(f"  Mean full cycle (incl wait):  {mean_cycle:.0f}ms")
    if failures:
        print("  Failure counts:")
        for k, v in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    else:
        print("  No endpoint failures recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference IO per Azure tick")
    parser.add_argument("--ticks", type=int, default=5, help="Number of tick cycles (default 5)")
    args = parser.parse_args()
    reports = run_benchmark(args.ticks)
    _summarize(reports)
    if any(r.skipped for r in reports) or any(r.over_budget for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
