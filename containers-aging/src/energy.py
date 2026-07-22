import asyncio
import glob
import os
import time

RAPL_BASE_PATH = os.environ.get("RAPL_BASE_PATH", "/sys/class/powercap")
THERMAL_BASE_PATH = os.environ.get("THERMAL_BASE_PATH", "/sys/class/thermal")

DEFAULT_RAPL_ZONES = ["package", "core", "uncore", "dram"]


def _read_int(path: str):
    try:
        with open(path, "r") as file:
            return int(file.read().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _read_str(path: str):
    try:
        with open(path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


class RaplReader:
    """
    Reads Intel RAPL energy counters directly from sysfs (no subprocess/sudo/bc involved).

    All zones are discovered at runtime instead of hardcoded (e.g. intel-rapl:0:1), so this
    keeps working across machines with a different RAPL layout (single/multi socket, with
    or without a dram/uncore domain).
    """

    def __init__(self, zone_filters=None):
        self.zones = self._discover_zones()
        self.zone_filters = list(zone_filters) if zone_filters else list(DEFAULT_RAPL_ZONES)
        self.selected_zones = self._select_zones(self.zone_filters)
        self.active_zone_names = sorted({name for names in self.selected_zones.values() for name in names})
        self.available = bool(self.selected_zones)

    def _discover_zones(self):
        zones = {}
        top_level_paths = sorted(glob.glob(f"{RAPL_BASE_PATH}/intel-rapl:*"))

        for zone_path in top_level_paths:
            zone_id = os.path.basename(zone_path)
            if ":" in zone_id[len("intel-rapl:"):]:
                continue  # this is a subzone (e.g. intel-rapl:0:1), handled below via its parent

            name = _read_str(os.path.join(zone_path, "name"))
            if name:
                zones[name] = {
                    "energy_path": os.path.join(zone_path, "energy_uj"),
                    "max_range_path": os.path.join(zone_path, "max_energy_range_uj"),
                }

            for child_path in sorted(glob.glob(f"{zone_path}/{zone_id}:*")):
                child_name = _read_str(os.path.join(child_path, "name"))
                if child_name:
                    zones[child_name] = {
                        "energy_path": os.path.join(child_path, "energy_uj"),
                        "max_range_path": os.path.join(child_path, "max_energy_range_uj"),
                    }

        return zones

    @staticmethod
    def _zone_matches_keyword(zone_name, keyword):
        """
        Exact match, or "keyword-N" (e.g. "package-0", "package-1" on multi-socket
        machines). A plain substring match would be wrong here: "core" is literally
        a substring of "uncore", so it would silently fold uncore's energy into core.
        """
        zone_name = zone_name.lower()
        keyword = keyword.lower()
        return zone_name == keyword or zone_name.startswith(f"{keyword}-")

    def _select_zones(self, filters):
        """Groups discovered zones by configured keyword (e.g. "package" -> ["package-0"])."""
        selected = {}
        for keyword in filters:
            matches = [name for name in self.zones if self._zone_matches_keyword(name, keyword)]
            if matches:
                selected[keyword] = matches
        return selected

    def read_energy_uj(self):
        """Reads energy_uj only for the selected zones (not every zone discovered)."""
        return {name: _read_int(self.zones[name]["energy_path"]) for name in self.active_zone_names}

    def _zone_delta_uj(self, before, after, zone_name):
        zone = self.zones.get(zone_name)
        value_before = before.get(zone_name)
        value_after = after.get(zone_name)

        if zone is None or value_before is None or value_after is None:
            return None

        if value_after >= value_before:
            return value_after - value_before

        # counter wrapped around, correct using the zone's max range
        max_range = _read_int(zone["max_range_path"])
        return None if max_range is None else (value_after + max_range) - value_before

    def group_delta_j(self, before, after, zone_names):
        """Sums the energy (in Joules) consumed by a group of zones (e.g. every package_zones)."""
        total = 0
        found_any = False

        for zone_name in zone_names:
            delta_uj = self._zone_delta_uj(before, after, zone_name)
            if delta_uj is not None:
                total += delta_uj
                found_any = True

        return (total / 1e6) if found_any else None


def find_pkg_temp_path():
    """
    Finds the CPU package thermal zone by its type ("x86_pkg_temp") instead of assuming
    a fixed thermal_zoneN index, which varies across machines.
    """
    for type_path in sorted(glob.glob(f"{THERMAL_BASE_PATH}/thermal_zone*/type")):
        if _read_str(type_path) == "x86_pkg_temp":
            return type_path.replace("/type", "/temp")
    return None


def read_cpu_temp_c(temp_path):
    if not temp_path:
        return None
    raw = _read_int(temp_path)
    return (raw / 1000) if raw is not None else None


def sample_power(reader: RaplReader, temp_path, interval):
    """
    Samples RAPL counters for `interval` seconds and returns the raw energy consumed (in
    Joules) by each selected domain (reader.zone_filters) during that window, plus the
    current CPU package temperature.

    These are raw, non-averaged values (matching the original collection script): no
    division by `interval` is applied. Only the configured domains are present in the
    returned dict (e.g. only "package_j" and "dram_j" if zone_filters was ["package",
    "dram"]). "total_j" is included whenever "package" and/or "dram" was selected, matching
    the original script's total = package + dram; it is omitted otherwise.
    """
    before = reader.read_energy_uj()
    time.sleep(interval)
    after = reader.read_energy_uj()

    result = {"cpu_temp_c": read_cpu_temp_c(temp_path)}

    for keyword, zone_names in reader.selected_zones.items():
        result[f"{keyword}_j"] = reader.group_delta_j(before, after, zone_names)

    package_j = result.get("package_j")
    dram_j = result.get("dram_j")
    if package_j is not None or dram_j is not None:
        result["total_j"] = (package_j or 0) + (dram_j or 0)

    return result


async def run_tomada_monitor(ip, username, password, interval, on_sample, on_error=None):
    """
    Connects to a TP-Link Kasa/Tapo smart plug and repeatedly samples its real-time power,
    invoking on_sample(dict) every `interval` seconds. Reconnects automatically if the
    connection is lost, so it can run unattended for the whole experiment.

    Requires the "python-kasa" package (only imported here, lazily, so the rest of the
    monitoring framework works fine without it when this feature is disabled).
    """
    from kasa import Discover

    device = None
    kwh_total = 0.0

    while True:
        try:
            if device is None:
                device = await Discover.discover_single(ip, username=username, password=password)

            await device.update()

            if not device.has_emeter:
                if on_error:
                    on_error("device has no energy meter")
                await asyncio.sleep(interval)
                continue

            emeter = await device.get_emeter_realtime()
            power_w = emeter.get("power_mw", 0) / 1000
            kwh_interval = power_w * interval / 3_600_000
            kwh_total += kwh_interval

            on_sample({
                "power_w": round(power_w, 3),
                "kwh_interval": round(kwh_interval, 6),
                "kwh_total": round(kwh_total, 6),
                "state": 1 if device.is_on else 0,
            })
        except Exception as error:
            if on_error:
                on_error(str(error))
            device = None  # force rediscovery on the next iteration

        await asyncio.sleep(interval)