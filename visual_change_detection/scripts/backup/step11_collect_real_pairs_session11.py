"""
step11_collect_real_pairs.py
============================
Phase A — Real Data Collection

Captures before/after screenshot pairs from two APK versions of Android apps
running on an emulator (or physical device) via ADB.

For each app in the registry, navigates to N configured screens and captures:
    {app}_{screen}_v1_original.jpg   — screenshot from version 1 (older)
    {app}_{screen}_v1_dump.xml       — UIAutomator layout dump from version 1
    {app}_{screen}_v2_changed.jpg    — screenshot from version 2 (newer)
    {app}_{screen}_v2_dump.xml       — UIAutomator layout dump from version 2

Feed these pairs into step12_annotate_ui_diff.py to generate GT annotations,
then into step13_build_real_dataset.py to produce a training/test split.

─── Prerequisites ──────────────────────────────────────────────────────────────
1. Android Studio installed (for AVD Manager + emulator):
       https://developer.android.com/studio

2. Create an AVD (Android Virtual Device):
       Android Studio → Device Manager → Create Device
       Recommended: Pixel 4, API 30 (Android 11), x86_64
       Resolution: 1080×2340 or 1080×1920

3. Start the emulator BEFORE running this script:
       %LOCALAPPDATA%\\Android\\Sdk\\emulator\\emulator.exe -avd <avd_name> -no-snapshot-load

4. ADB in your PATH (comes with Android Studio):
       Default: %LOCALAPPDATA%\\Android\\Sdk\\platform-tools\\adb.exe
       Add to PATH or set --adb_path argument.

5. Check connected devices: python scripts/step11_collect_real_pairs.py --list_devices

─── Usage ──────────────────────────────────────────────────────────────────────
    # List connected devices / emulators
    python scripts/step11_collect_real_pairs.py --list_devices

    # Download APKs for one app from F-Droid (auto selects last 2 releases)
    python scripts/step11_collect_real_pairs.py --download --package de.danoeh.antennapod --apk_dir ./real_data/apks

    # Download APKs for ALL apps in the registry
    python scripts/step11_collect_real_pairs.py --download_all --registry ./real_data/app_registry.json --apk_dir ./real_data/apks

    # Collect screen pairs for one app (APKs already downloaded)
    python scripts/step11_collect_real_pairs.py --app AntennaPod --registry ./real_data/app_registry.json --v1_apk ./real_data/apks/AntennaPod_v1.apk --v2_apk ./real_data/apks/AntennaPod_v2.apk --device emulator-5554 --output_dir ./real_data/raw_pairs

    # Collect pairs for ALL apps whose APKs are in apk_dir
    python scripts/step11_collect_real_pairs.py --collect_all --registry ./real_data/app_registry.json --apk_dir ./real_data/apks --device emulator-5554 --output_dir ./real_data/raw_pairs
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

# Force UTF-8 stdout so Unicode chars in print() don't crash on Windows cmd
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─── ADB wait times ──────────────────────────────────────────────────────────
WAIT_APP_LAUNCH   = 4.0   # seconds after launching app before first screenshot
WAIT_AFTER_ACTION = 2.0   # seconds after each navigation action
WAIT_UI_DUMP      = 3.0   # seconds after requesting UI dump (dump can be slow)

FDROID_API_BASE   = "https://f-droid.org/api/v1/packages"
FDROID_REPO_BASE  = "https://f-droid.org/repo"


# =============================================================================
# ADB HELPERS
# =============================================================================

def run_adb(adb_path, device_id, *args, timeout=30, check=False):
    """
    Run an ADB command targeting a specific device.
    Returns (returncode, stdout_str, stderr_str).
    """
    cmd = [adb_path, "-s", device_id] + list(args)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=check
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"[WARN] ADB command timed out: {' '.join(args)}")
        return -1, "", "timeout"
    except FileNotFoundError:
        print(f"[ERROR] ADB not found at: {adb_path}")
        print("        Install Android Studio or add platform-tools to PATH.")
        sys.exit(1)


def list_devices(adb_path):
    """Print all connected ADB devices/emulators and return their IDs."""
    result = subprocess.run(
        [adb_path, "devices", "-l"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.strip().splitlines()
    devices = []
    for line in lines[1:]:                     # skip "List of devices attached"
        if "\t" in line and "offline" not in line:
            device_id = line.split("\t")[0].strip()
            devices.append(device_id)
            print(f"  {line}")
    return devices


def install_apk(adb_path, device_id, apk_path):
    """Install APK, replacing any existing version. Returns True on success."""
    print(f"[ADB] Installing {Path(apk_path).name} ...")
    rc, out, err = run_adb(adb_path, device_id, "install", "-r", "-t", str(apk_path), timeout=120)
    if rc != 0 or "Failure" in out or "Exception" in out:
        print(f"[WARN] Install may have failed: {out or err}")
        return False
    print(f"[ADB] Install OK")
    return True


def uninstall_package(adb_path, device_id, package_name):
    """Uninstall a package (ignores error if not installed)."""
    run_adb(adb_path, device_id, "uninstall", package_name, timeout=30)


def launch_activity(adb_path, device_id, package_name, activity=None):
    """Launch app main activity (or a specific activity if given)."""
    if activity:
        component = f"{package_name}/{activity}"
    else:
        component = None

    if component:
        rc, out, err = run_adb(
            adb_path, device_id,
            "shell", "am", "start", "-n", component
        )
    else:
        rc, out, err = run_adb(
            adb_path, device_id,
            "shell", "monkey", "-p", package_name,
            "--pct-syskeys", "0", "-v", "1"
        )
    return rc == 0


def press_back(adb_path, device_id):
    run_adb(adb_path, device_id, "shell", "input", "keyevent", "KEYCODE_BACK")


def press_home(adb_path, device_id):
    run_adb(adb_path, device_id, "shell", "input", "keyevent", "KEYCODE_HOME")


def press_key(adb_path, device_id, keycode):
    run_adb(adb_path, device_id, "shell", "input", "keyevent", keycode)


def tap(adb_path, device_id, x, y):
    run_adb(adb_path, device_id, "shell", "input", "tap", str(int(x)), str(int(y)))


def get_screenshot(adb_path, device_id, local_path):
    """Capture screenshot via adb screencap + pull. Returns True on success."""
    remote = "/sdcard/_ss.png"
    rc1, _, _ = run_adb(adb_path, device_id, "shell", "screencap", "-p", remote)
    rc2, _, _ = run_adb(adb_path, device_id, "pull", remote, str(local_path), timeout=20)
    return rc1 == 0 and rc2 == 0


def get_ui_dump(adb_path, device_id, local_path):
    """Capture UIAutomator layout dump. Returns True on success."""
    remote = "/sdcard/_uidump.xml"
    rc1, out1, _ = run_adb(adb_path, device_id, "shell", "uiautomator", "dump", remote, timeout=30)
    if rc1 != 0 or "ERROR" in out1.upper():
        print(f"[WARN] UI dump failed: {out1}")
        return False
    time.sleep(WAIT_UI_DUMP)
    rc2, _, _ = run_adb(adb_path, device_id, "pull", remote, str(local_path), timeout=20)
    return rc2 == 0


def find_element_in_dump(xml_path, resource_id=None, text=None, content_desc=None):
    """
    Parse a UIAutomator XML dump and find an element by resource-id, text,
    or content-desc. Returns the element's centre (cx, cy) or None.
    """
    if not Path(xml_path).exists():
        return None
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError:
        return None

    def iter_nodes(node):
        yield node
        for child in node:
            yield from iter_nodes(child)

    for node in iter_nodes(root):
        attrs = node.attrib
        match = False
        if resource_id and attrs.get("resource-id", "") == resource_id:
            match = True
        elif text and attrs.get("text", "").lower() == text.lower():
            match = True
        elif content_desc and content_desc.lower() in attrs.get("content-desc", "").lower():
            match = True

        if match:
            bounds_str = attrs.get("bounds", "")
            bounds = _parse_bounds(bounds_str)
            if bounds:
                x1, y1, x2, y2 = bounds
                return (x1 + x2) // 2, (y1 + y2) // 2
    return None


def _parse_bounds(bounds_str):
    """Parse UIAutomator bounds '[x1,y1][x2,y2]' → (x1,y1,x2,y2) or None."""
    try:
        parts = bounds_str.replace("][", ",").strip("[]").split(",")
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return None


# =============================================================================
# NAVIGATION ACTION EXECUTOR
# =============================================================================

def execute_action(adb_path, device_id, action, dump_path=None):
    """
    Execute a single navigation action dict.

    Supported action types:
        tap_coord        — tap at fixed coordinates {x, y}
        tap_resource_id  — find element by resource-id in dump, tap its centre
        tap_text         — find element by text in dump, tap its centre
        tap_content_desc — find element by content-desc in dump, tap centre
        press_key        — send a keyevent {keycode}
        wait             — sleep {ms} milliseconds
        press_back
        press_home

    Falls back to fallback_coord if element not found in dump.
    """
    atype = action.get("type", "")

    if atype == "tap_coord":
        tap(adb_path, device_id, action["x"], action["y"])

    elif atype == "tap_resource_id":
        coord = None
        if dump_path:
            coord = find_element_in_dump(dump_path, resource_id=action.get("resource_id"))
        if coord is None:
            fb = action.get("fallback_coord")
            if fb:
                coord = fb
        if coord:
            tap(adb_path, device_id, coord[0], coord[1])
        else:
            print(f"[WARN] Element not found: resource_id={action.get('resource_id')}")

    elif atype == "tap_text":
        coord = None
        if dump_path:
            coord = find_element_in_dump(dump_path, text=action.get("text"))
        if coord is None:
            fb = action.get("fallback_coord")
            if fb:
                coord = fb
        if coord:
            tap(adb_path, device_id, coord[0], coord[1])
        else:
            print(f"[WARN] Element not found: text='{action.get('text')}'")

    elif atype == "tap_content_desc":
        coord = None
        if dump_path:
            coord = find_element_in_dump(dump_path, content_desc=action.get("content_desc"))
        if coord is None:
            fb = action.get("fallback_coord")
            if fb:
                coord = fb
        if coord:
            tap(adb_path, device_id, coord[0], coord[1])
        else:
            print(f"[WARN] Element not found: content_desc='{action.get('content_desc')}'")

    elif atype == "press_key":
        press_key(adb_path, device_id, action.get("keycode", "KEYCODE_MENU"))

    elif atype == "press_back":
        press_back(adb_path, device_id)

    elif atype == "press_home":
        press_home(adb_path, device_id)

    elif atype == "wait":
        time.sleep(action.get("ms", 1000) / 1000.0)

    else:
        print(f"[WARN] Unknown action type: {atype}")

    time.sleep(WAIT_AFTER_ACTION)


def capture_screen_state(adb_path, device_id, pair_id, output_dir):
    """
    Capture screenshot + UI dump for the current screen state.
    Returns (screenshot_path, dump_path) or (None, None) on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ss_path   = output_dir / f"{pair_id}.jpg"
    dump_path = output_dir / f"{pair_id}_dump.xml"

    ok_ss = get_screenshot(adb_path, device_id, ss_path)
    if not ok_ss:
        print(f"[ERROR] Screenshot failed for {pair_id}")
        return None, None

    ok_dump = get_ui_dump(adb_path, device_id, dump_path)
    if not ok_dump:
        print(f"[WARN] UI dump failed for {pair_id} — screenshot kept, dump missing")
        dump_path = None

    return ss_path, dump_path


# =============================================================================
# PER-APP COLLECTION
# =============================================================================

def collect_one_version(adb_path, device_id, app_cfg, version_label, output_dir):
    """
    Navigate through all configured screens for one version of an app and
    capture screenshot + UI dump for each.

    Returns list of {screen_id, screenshot_path, dump_path}.
    """
    package  = app_cfg["package"]
    activity = app_cfg.get("main_activity")
    screens  = app_cfg.get("screens", [])
    app_name = app_cfg["name"].lower().replace(" ", "_")

    results = []
    print(f"\n[COLLECT] {app_cfg['name']} — version: {version_label}")

    for screen_cfg in screens:
        screen_id = screen_cfg["screen_id"]
        actions   = screen_cfg.get("actions", [])
        pair_id   = f"{app_name}_{screen_id}_{version_label}"

        print(f"  [SCREEN] {screen_id}")

        # Return to app home between screens
        press_home(adb_path, device_id)
        time.sleep(0.5)
        launch_activity(adb_path, device_id, package, activity)
        time.sleep(WAIT_APP_LAUNCH)

        # Capture an initial dump for find_element_in_dump lookups during navigation
        tmp_dump = output_dir / f"_nav_dump.xml"
        get_ui_dump(adb_path, device_id, tmp_dump)

        # Execute navigation actions
        for action in actions:
            execute_action(adb_path, device_id, action, dump_path=tmp_dump)
            # Refresh dump after each action
            get_ui_dump(adb_path, device_id, tmp_dump)

        # Capture final state
        ss_path, dump_path = capture_screen_state(
            adb_path, device_id, pair_id, output_dir
        )

        if ss_path:
            results.append({
                "screen_id":       screen_id,
                "pair_id":         pair_id,
                "screenshot_path": str(ss_path),
                "dump_path":       str(dump_path) if dump_path else None,
            })
            print(f"    ✓ Captured: {pair_id}.jpg")
        else:
            print(f"    ✗ Failed: {pair_id}")

    # Clean up temp dump
    if tmp_dump.exists():
        tmp_dump.unlink()

    return results


def collect_app_pairs(adb_path, device_id, app_cfg,
                      v1_apk, v2_apk, output_dir):
    """
    Install v1, collect all screens, uninstall.
    Install v2, collect all screens, uninstall.
    Returns list of matched pairs ready for annotation.
    """
    package  = app_cfg["package"]
    app_name = app_cfg["name"].lower().replace(" ", "_")
    raw_dir  = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Version 1 ──────────────────────────────────────────────────────────────
    uninstall_package(adb_path, device_id, package)
    ok = install_apk(adb_path, device_id, v1_apk)
    if not ok:
        print(f"[ERROR] Cannot install v1 for {app_cfg['name']}. Skipping.")
        return []
    v1_results = collect_one_version(adb_path, device_id, app_cfg, "v1", raw_dir)
    uninstall_package(adb_path, device_id, package)

    # ── Version 2 ──────────────────────────────────────────────────────────────
    ok = install_apk(adb_path, device_id, v2_apk)
    if not ok:
        print(f"[ERROR] Cannot install v2 for {app_cfg['name']}. Skipping.")
        return []
    v2_results = collect_one_version(adb_path, device_id, app_cfg, "v2", raw_dir)
    uninstall_package(adb_path, device_id, package)

    # ── Match screens by screen_id ─────────────────────────────────────────────
    v1_by_screen = {r["screen_id"]: r for r in v1_results}
    v2_by_screen = {r["screen_id"]: r for r in v2_results}

    pairs_dir = Path(output_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    matched_pairs = []
    for screen_id in v1_by_screen:
        if screen_id not in v2_by_screen:
            print(f"[WARN] Screen '{screen_id}' missing from v2. Skipping.")
            continue

        r1 = v1_by_screen[screen_id]
        r2 = v2_by_screen[screen_id]
        pair_id = f"{app_name}_{screen_id}"

        # Copy as original/changed using the pipeline's naming convention
        orig_dst = pairs_dir / f"{pair_id}_original.jpg"
        chng_dst = pairs_dir / f"{pair_id}_changed.jpg"
        shutil.copy2(r1["screenshot_path"], orig_dst)
        shutil.copy2(r2["screenshot_path"], chng_dst)

        d1_dst, d2_dst = None, None
        if r1["dump_path"]:
            d1_dst = pairs_dir / f"{pair_id}_v1_dump.xml"
            shutil.copy2(r1["dump_path"], d1_dst)
        if r2["dump_path"]:
            d2_dst = pairs_dir / f"{pair_id}_v2_dump.xml"
            shutil.copy2(r2["dump_path"], d2_dst)

        matched_pairs.append({
            "pair_id":       pair_id,
            "app":           app_cfg["name"],
            "screen_id":     screen_id,
            "original_path": str(orig_dst),
            "changed_path":  str(chng_dst),
            "v1_dump":       str(d1_dst) if d1_dst else None,
            "v2_dump":       str(d2_dst) if d2_dst else None,
        })

    print(f"\n[COLLECT] {app_cfg['name']}: {len(matched_pairs)} pairs collected → {pairs_dir}")
    return matched_pairs


# =============================================================================
# F-DROID APK DOWNLOADER
# =============================================================================

def fetch_fdroid_versions(package_name):
    """
    Fetch version list from F-Droid API.
    Returns list of version dicts sorted oldest → newest (by versionCode).
    """
    url = f"{FDROID_API_BASE}/{package_name}"
    print(f"[FDROID] Fetching version list: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/step11"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[ERROR] F-Droid API error {e.code} for {package_name}")
        return []
    except Exception as e:
        print(f"[ERROR] F-Droid API unreachable: {e}")
        return []

    versions = data.get("packages", [])
    versions.sort(key=lambda v: v.get("versionCode", 0))
    return versions


def download_fdroid_apk(package_name, version_info, output_dir):
    """
    Download a single APK from F-Droid repo. Returns local path or None.

    F-Droid API v1 often omits the 'apkName' field — when absent we construct
    it using the standard repo naming convention:
        {packageName}_{versionCode}.apk
    """
    apk_name = version_info.get("apkName")
    if not apk_name:
        version_code = version_info.get("versionCode")
        if not version_code:
            print(f"[WARN] Cannot determine APK name — no apkName or versionCode: {version_info}")
            return None
        apk_name = f"{package_name}_{version_code}.apk"
        print(f"[FDROID] Constructed APK name: {apk_name}")

    url       = f"{FDROID_REPO_BASE}/{apk_name}"
    out_path  = Path(output_dir) / apk_name
    if out_path.exists():
        print(f"[FDROID] Already downloaded: {apk_name}")
        return str(out_path)

    print(f"[FDROID] Downloading {apk_name} ({version_info.get('versionName', '?')}) ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/step11"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)
        print(f"\r  Done.{' ' * 30}")
        return str(out_path)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def download_direct_apk(url, output_dir, label):
    """
    Download an APK from a direct URL (e.g. GitHub releases).
    Saves as {label}.apk — e.g. 'AntennaPod_v1.apk'.
    Returns local path or None.
    """
    out_path = Path(output_dir) / f"{label}.apk"
    if out_path.exists():
        print(f"[DIRECT] Already downloaded: {out_path.name}")
        return str(out_path)

    print(f"[DIRECT] Downloading {label} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 Python/step11"
        })
        with urllib.request.urlopen(req, timeout=180) as resp, \
             open(out_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/total*100:.1f}%  "
                          f"({downloaded//1024//1024} MB)", end="", flush=True)
        print(f"\r  Done.{' '*30}")
        return str(out_path)
    except Exception as e:
        print(f"[ERROR] Direct download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def download_app_apks(app_cfg, output_dir):
    """
    Download two APK versions for an app.

    Prefers direct URLs (v1_url / v2_url in app_cfg) over F-Droid API.
    Direct URLs point to GitHub releases or other hosting with significantly
    different UI versions.  F-Droid API only keeps recent patch releases.

    Returns (v1_apk_path, v2_apk_path) or (None, None) on failure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    app_name = app_cfg["name"].replace(" ", "")

    # ── Direct URL download (preferred — allows major-version gaps) ──────────
    v1_url = app_cfg.get("v1_url")
    v2_url = app_cfg.get("v2_url")

    if v1_url and v2_url:
        v1_path = download_direct_apk(v1_url, output_dir, f"{app_name}_v1")
        v2_path = download_direct_apk(v2_url, output_dir, f"{app_name}_v2")
        if v1_path and v2_path:
            return v1_path, v2_path
        print(f"[WARN] Direct URL download failed for {app_cfg['name']}. "
              f"Falling back to F-Droid.")

    # ── F-Droid API fallback ─────────────────────────────────────────────────
    package = app_cfg.get("fdroid_package", app_cfg.get("package"))
    v1_idx  = app_cfg.get("older_version_idx", -3)
    v2_idx  = app_cfg.get("newer_version_idx", -1)

    versions = fetch_fdroid_versions(package)
    if len(versions) < 2:
        print(f"[WARN] Not enough versions found for {package}: {len(versions)}")
        return None, None

    try:
        v1_info = versions[v1_idx]
        v2_info = versions[v2_idx]
    except IndexError:
        v1_info, v2_info = versions[-2], versions[-1]

    print(f"[FDROID] {app_cfg['name']}: "
          f"v1={v1_info.get('versionName', '?')} -> v2={v2_info.get('versionName', '?')}")

    v1_path = download_fdroid_apk(package, v1_info, output_dir)
    v2_path = download_fdroid_apk(package, v2_info, output_dir)
    return v1_path, v2_path


# =============================================================================
# MANIFEST
# =============================================================================

def save_collection_manifest(all_pairs, output_dir):
    """Save a JSON manifest of all collected raw pairs."""
    manifest = {
        "total_pairs":  len(all_pairs),
        "pairs":        all_pairs,
    }
    path = Path(output_dir) / "collection_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[INFO] Collection manifest saved: {path}")
    print(f"[INFO] Total pairs collected: {len(all_pairs)}")
    return str(path)


# =============================================================================
# MAIN
# =============================================================================

def load_registry(registry_path):
    with open(registry_path) as f:
        return json.load(f)


def _apk_version_code(path):
    """
    Extract the numeric versionCode from an F-Droid APK filename.
    e.g. 'de.danoeh.antennapod_3110095.apk' → 3110095
    Falls back to 0 if the suffix is not a pure integer.
    """
    stem = Path(path).stem                    # strip .apk
    parts = stem.rsplit("_", 1)               # split on last underscore
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def find_apk_for_app(app_cfg, apk_dir, version_label):
    """
    Find the correct v1 (older) or v2 (newer) APK for an app in apk_dir.

    Three strategies tried in order:

    1. Manual naming convention: {AppName}_v1.apk / {AppName}_v2.apk
       (for APKs the user placed manually)

    2. F-Droid downloaded convention: {package}_{versionCode}.apk
       Scans for all files starting with the package name, sorts by the
       numeric versionCode suffix, and returns the oldest (v1) or newest (v2).

    3. Package-fragment fuzzy match: any .apk whose name contains the last
       component of the package name (e.g. 'antennapod'), sorted the same way.
    """
    apk_dir = Path(apk_dir)
    package  = app_cfg["package"]

    # ── Strategy 1: manual {AppName}_v1.apk ──────────────────────────────────
    app_name  = app_cfg["name"].replace(" ", "")
    candidate = apk_dir / f"{app_name}_{version_label}.apk"
    if candidate.exists():
        return str(candidate)

    # ── Strategy 2: F-Droid {package}_{versionCode}.apk ──────────────────────
    fdroid_apks = sorted(
        apk_dir.glob(f"{package}_*.apk"),
        key=_apk_version_code
    )
    if len(fdroid_apks) >= 2:
        chosen = fdroid_apks[0] if version_label == "v1" else fdroid_apks[-1]
        return str(chosen)
    if len(fdroid_apks) == 1:
        # Only one copy present — usable for whichever label is requested but
        # both v1 and v2 will point to the same file; caller handles this.
        return str(fdroid_apks[0])

    # ── Strategy 3: package-fragment fuzzy match ──────────────────────────────
    pkg_frag  = package.split(".")[-1].lower()
    fuzzy     = sorted(
        [f for f in apk_dir.glob("*.apk") if pkg_frag in f.name.lower()],
        key=_apk_version_code
    )
    if len(fuzzy) >= 2:
        chosen = fuzzy[0] if version_label == "v1" else fuzzy[-1]
        return str(chosen)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Phase A — Real paired screenshot collection via ADB"
    )

    parser.add_argument("--adb_path",   default="adb",
                        help="Path to adb executable (default: 'adb' from PATH)")
    parser.add_argument("--registry",   default="./real_data/app_registry.json",
                        help="App registry JSON (default: ./real_data/app_registry.json)")
    parser.add_argument("--device",     default=None,
                        help="ADB device ID (e.g. emulator-5554). Auto-detected if only one device.")
    parser.add_argument("--output_dir", default="./real_data",
                        help="Root output directory (default: ./real_data)")

    # ── Mode flags ────────────────────────────────────────────────────────────
    modes = parser.add_argument_group("Modes (choose one)")
    modes.add_argument("--list_devices",  action="store_true",
                       help="List connected ADB devices and exit")
    modes.add_argument("--download",      action="store_true",
                       help="Download APKs for one app from F-Droid")
    modes.add_argument("--download_all",  action="store_true",
                       help="Download APKs for ALL apps in registry")
    modes.add_argument("--collect",       action="store_true",
                       help="Collect screen pairs for one app (APKs pre-downloaded)")
    modes.add_argument("--collect_all",   action="store_true",
                       help="Collect pairs for all apps whose APKs are in apk_dir")

    # ── Per-app arguments ─────────────────────────────────────────────────────
    parser.add_argument("--app",      default=None, help="App name from registry (for --collect)")
    parser.add_argument("--v1_apk",   default=None, help="Path to v1 APK (for --collect)")
    parser.add_argument("--v2_apk",   default=None, help="Path to v2 APK (for --collect)")
    parser.add_argument("--package",  default=None, help="Package name (for --download)")
    parser.add_argument("--apk_dir",  default="./real_data/apks",
                        help="APK storage directory (default: ./real_data/apks)")

    args = parser.parse_args()

    # ── List devices ──────────────────────────────────────────────────────────
    if args.list_devices:
        print("[INFO] Connected ADB devices:")
        devs = list_devices(args.adb_path)
        if not devs:
            print("  (none found — is the emulator running?)")
        return

    # ── Resolve device ID (skip ADB probe for download-only operations) ──────
    device_id = args.device
    if not (args.download or args.download_all):
        if not device_id:
            devs = list_devices(args.adb_path)
            if len(devs) == 1:
                device_id = devs[0]
                print(f"[INFO] Auto-detected device: {device_id}")
            elif len(devs) > 1:
                print(f"[ERROR] Multiple devices found. Specify --device: {devs}")
                sys.exit(1)
            else:
                print("[ERROR] No ADB device found. Start your emulator first.")
                sys.exit(1)

    # ── Download one app ──────────────────────────────────────────────────────
    if args.download:
        if not args.package:
            print("[ERROR] --package required with --download")
            sys.exit(1)
        registry = load_registry(args.registry)
        cfg = next((a for a in registry["apps"] if a["package"] == args.package), None)
        if cfg is None:
            # Build minimal cfg from package alone
            cfg = {"name": args.package, "package": args.package,
                   "fdroid_package": args.package,
                   "older_version_idx": -3, "newer_version_idx": -1}
        v1, v2 = download_app_apks(cfg, args.apk_dir)
        if v1 and v2:
            print(f"[OK] v1: {v1}")
            print(f"[OK] v2: {v2}")
        return

    # ── Download all apps ─────────────────────────────────────────────────────
    if args.download_all:
        registry = load_registry(args.registry)
        for app_cfg in registry["apps"]:
            v1, v2 = download_app_apks(app_cfg, args.apk_dir)
            if not v1 or not v2:
                print(f"[WARN] Skipping {app_cfg['name']} — APK download failed")
        print("\n[INFO] All downloads complete.")
        return

    # ── Collect one app ───────────────────────────────────────────────────────
    if args.collect:
        if not args.app or not args.v1_apk or not args.v2_apk:
            print("[ERROR] --app, --v1_apk, and --v2_apk required with --collect")
            sys.exit(1)
        registry = load_registry(args.registry)
        app_cfg = next((a for a in registry["apps"] if a["name"] == args.app), None)
        if app_cfg is None:
            print(f"[ERROR] App '{args.app}' not found in registry.")
            sys.exit(1)
        pairs = collect_app_pairs(
            args.adb_path, device_id, app_cfg,
            args.v1_apk, args.v2_apk,
            Path(args.output_dir) / app_cfg["name"].lower().replace(" ", "_")
        )
        if pairs:
            save_collection_manifest(pairs, args.output_dir)
        return

    # ── Collect all apps ──────────────────────────────────────────────────────
    if args.collect_all:
        registry  = load_registry(args.registry)
        all_pairs = []
        for app_cfg in registry["apps"]:
            v1 = find_apk_for_app(app_cfg, args.apk_dir, "v1")
            v2 = find_apk_for_app(app_cfg, args.apk_dir, "v2")
            if not v1 or not v2:
                print(f"[WARN] APKs not found for {app_cfg['name']} — run --download_all first")
                continue
            pairs = collect_app_pairs(
                args.adb_path, device_id, app_cfg,
                v1, v2,
                Path(args.output_dir) / app_cfg["name"].lower().replace(" ", "_")
            )
            all_pairs.extend(pairs)
        save_collection_manifest(all_pairs, args.output_dir)
        print(f"\n[DONE] {len(all_pairs)} pairs collected across {len(registry['apps'])} apps.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
