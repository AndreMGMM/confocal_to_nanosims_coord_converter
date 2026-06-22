
from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageSequence
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="Overview to NanoSIMS",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --panel: #121722;
        --panel-2: #181f2e;
        --border: #2a3448;
        --text-soft: #9eabc2;
        --accent: #4f8cff;
    }
    .stApp {
        background:
          radial-gradient(circle at top left, rgba(79,140,255,.12), transparent 32rem),
          #0c1018;
    }
    [data-testid="stSidebar"] {
        background: #101622;
        border-right: 1px solid var(--border);
    }
    [data-testid="stHeader"] {
        background: rgba(12,16,24,.75);
    }
    .block-container {
        max-width: 1800px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin-bottom: .1rem;
    }
    .app-subtitle {
        color: var(--text-soft);
        margin-bottom: 1rem;
    }
    .status-box, .metric-card {
        background: linear-gradient(180deg, #171e2c, #121824);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .8rem 1rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #34415a;
        background: #192235;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


LUTS = {
    "Green": (0.0, 1.0, 0.0),
    "Red": (1.0, 0.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "White": (1.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
}


def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 3 or src.shape[1] != 2:
        raise ValueError("At least three matching 2D source and destination points are required.")
    n = src.shape[0]
    x = np.zeros((2 * n, 6), dtype=float)
    b = np.zeros((2 * n,), dtype=float)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
        x[2 * i, :3] = [sx, sy, 1.0]
        x[2 * i + 1, 3:] = [sx, sy, 1.0]
        b[2 * i] = dx
        b[2 * i + 1] = dy
    a, *_ = np.linalg.lstsq(x, b, rcond=None)
    return np.array([[a[0], a[1], a[2]], [a[3], a[4], a[5]]], dtype=float)


def apply_affine(a: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y = xy
    return (
        float(a[0, 0] * x + a[0, 1] * y + a[0, 2]),
        float(a[1, 0] * x + a[1, 1] * y + a[1, 2]),
    )


def invert_affine(a: np.ndarray) -> np.ndarray:
    full = np.array(
        [[a[0, 0], a[0, 1], a[0, 2]], [a[1, 0], a[1, 1], a[1, 2]], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return np.linalg.inv(full)[:2, :]


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def image_to_channel_arrays(raw: bytes, max_channels: int = 2) -> List[np.ndarray]:
    img = Image.open(io.BytesIO(raw))
    channels: List[np.ndarray] = []
    for frame in ImageSequence.Iterator(img):
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            for c in range(min(arr.shape[-1], max_channels - len(channels))):
                channels.append(arr[..., c].astype(float))
                if len(channels) >= max_channels:
                    return channels
        else:
            channels.append(arr.astype(float))
            if len(channels) >= max_channels:
                return channels
    return channels


def composite_to_rgb(raw: bytes, settings: List[dict]) -> Image.Image:
    channels = image_to_channel_arrays(raw, max_channels=2)
    if not channels:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    h, w = channels[0].shape[:2]
    out = np.zeros((h, w, 3), dtype=float)
    for i, arr in enumerate(channels[:2]):
        cfg = settings[i]
        if not cfg["enabled"]:
            continue
        lo, hi = float(cfg["min"]), float(cfg["max"])
        norm = np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1)
        lut = LUTS[cfg["lut"]]
        for c in range(3):
            out[..., c] += norm * lut[c]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")


def normalized_name(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1].lower()


def load_bundle(uploaded) -> Tuple[dict, Dict[str, bytes], str]:
    files: Dict[str, bytes] = {}
    json_name = ""
    if uploaded.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                raw = zf.read(info.filename)
                files[normalized_name(info.filename)] = raw
                if info.filename.lower().endswith(".json") and not json_name:
                    json_name = normalized_name(info.filename)
        if not json_name:
            raise ValueError("The ZIP file does not contain a JSON mapping file.")
        data = json.loads(files[json_name].decode("utf-8"))
    else:
        json_name = uploaded.name
        data = json.loads(uploaded.getvalue().decode("utf-8"))
    return data, files, json_name


def initial_channel_settings(data: dict, files: Dict[str, bytes]) -> List[dict]:
    mins, maxs = [[], []], [[], []]
    for tile in (data.get("mosaic") or {}).get("tiles") or []:
        raw_path = str(tile.get("path") or "")
        raw = files.get(normalized_name(raw_path))
        if raw is None:
            continue
        try:
            channels = image_to_channel_arrays(raw, 2)
            for i, arr in enumerate(channels[:2]):
                mins[i].append(float(np.nanpercentile(arr, 1)))
                maxs[i].append(float(np.nanpercentile(arr, 99.8)))
        except Exception:
            continue
    result = []
    for i, lut in enumerate(("Green", "Red")):
        lo = min(mins[i]) if mins[i] else 0.0
        hi = max(maxs[i]) if maxs[i] else 1.0
        result.append({"enabled": True, "min": lo, "max": max(hi, lo + 1.0), "lut": lut})
    return result


def get_affines(data: dict):
    raw = data.get("affine_A")
    if raw is None:
        return None, None
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (2, 3):
        return None, None
    return arr, invert_affine(arr)


def get_positions(data: dict) -> Dict[str, dict]:
    positions = {}
    entries = (data.get("positions") or {}).get("entries") or {}
    for pid, entry in entries.items():
        coarse = entry.get("coarse_off") or {}
        try:
            stage = (float(coarse["x"]), float(coarse["y"]))
            z = float(coarse.get("z", 0.0))
        except Exception:
            stage, z = None, None
        positions[str(pid)] = {"stage": stage, "z": z, "nano": None}
    return positions


def build_overview(data: dict, files: Dict[str, bytes], settings: List[dict]) -> Tuple[Image.Image, int, int]:
    mosaic = data.get("mosaic") or {}
    size = mosaic.get("size") or [1200, 900]
    width, height = int(size[0]), int(size[1])
    overview = Image.new("RGB", (max(width, 1), max(height, 1)), (18, 21, 29))
    loaded = missing = 0
    for tile in mosaic.get("tiles") or []:
        raw_path = str(tile.get("path") or "")
        raw = files.get(normalized_name(raw_path))
        if raw is None:
            missing += 1
            continue
        try:
            image = composite_to_rgb(raw, settings)
            ox = int(tile.get("offset_x", tile.get("ox", 0)))
            oy = int(tile.get("offset_y", tile.get("oy", 0)))
            overview.paste(image, (ox, oy))
            loaded += 1
        except Exception:
            missing += 1
    if loaded == 0:
        draw = ImageDraw.Draw(overview)
        draw.text((25, 25), "No image tiles found. Upload a ZIP containing the JSON and tile images.", fill=(220, 225, 235))
    return overview, loaded, missing


def draw_overlay(image: Image.Image, positions: Dict[str, dict], stage_to_mosaic, anchors: List[dict]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if stage_to_mosaic is not None:
        for pid, pos in positions.items():
            if pos["stage"] is None:
                continue
            x, y = apply_affine(stage_to_mosaic, pos["stage"])
            r = 5
            draw.ellipse((x-r, y-r, x+r, y+r), outline=(80, 213, 255), width=2)
            draw.text((x+8, y-12), pid, fill=(230, 250, 255))
    for i, anchor in enumerate(anchors, 1):
        x, y = anchor["mosaic"]
        r = 8
        draw.line((x-r, y, x+r, y), fill=(255, 207, 74), width=3)
        draw.line((x, y-r, x, y+r), fill=(255, 207, 74), width=3)
        draw.text((x+10, y+5), f"A{i}", fill=(255, 235, 156))
    return out


def recompute(positions: Dict[str, dict], anchors: List[dict]):
    if len(anchors) < 3:
        for pos in positions.values():
            pos["nano"] = None
        return None
    src = np.asarray([a["stage"] for a in anchors], dtype=float)
    dst = np.asarray([a["nanosims"] for a in anchors], dtype=float)
    matrix = affine_from_points(src, dst)
    for pos in positions.values():
        pos["nano"] = apply_affine(matrix, pos["stage"]) if pos["stage"] is not None else None
    return matrix


def positions_df(positions: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for pid in sorted(positions):
        pos = positions[pid]
        stage, nano = pos["stage"], pos["nano"]
        rows.append({
            "ID": pid,
            "Confocal X (m)": stage[0] if stage else np.nan,
            "Confocal Y (m)": stage[1] if stage else np.nan,
            "Z (m)": pos["z"] if pos["z"] is not None else np.nan,
            "NanoSIMS X": nano[0] if nano else np.nan,
            "NanoSIMS Y": nano[1] if nano else np.nan,
        })
    return pd.DataFrame(rows)


def export_json(data: dict, positions: Dict[str, dict], anchors: List[dict], matrix) -> bytes:
    result = copy.deepcopy(data)
    result["nanosims_conversion"] = {
        "anchors": [{
            "mosaic": [float(a["mosaic"][0]), float(a["mosaic"][1])],
            "confocal_stage": [float(a["stage"][0]), float(a["stage"][1])],
            "nanosims": [float(a["nanosims"][0]), float(a["nanosims"][1])],
        } for a in anchors],
        "stage_to_nanosims_A": matrix.tolist() if matrix is not None else None,
    }
    entries = ((result.get("positions") or {}).get("entries") or {})
    for pid, pos in positions.items():
        if pid not in entries:
            continue
        nano = pos["nano"]
        if nano is None:
            entries[pid].pop("nanosims_off", None)
        else:
            entries[pid]["nanosims_off"] = {"x": float(nano[0]), "y": float(nano[1])}
    return json.dumps(result, indent=2).encode("utf-8")


def initialize_state(data, files, json_name):
    st.session_state.data = data
    st.session_state.files = files
    st.session_state.json_name = json_name
    st.session_state.settings = initial_channel_settings(data, files)
    st.session_state.positions = get_positions(data)
    st.session_state.stage_to_mosaic, st.session_state.mosaic_to_stage = get_affines(data)
    st.session_state.anchors = []
    st.session_state.matrix = None
    st.session_state.pending_click = None


st.markdown('<div class="app-title">Overview to NanoSIMS Converter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Browser-based mapping, anchor fitting and coordinate export.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Project")
    upload = st.file_uploader(
        "Mapping JSON or project ZIP",
        type=["json", "zip"],
        help="A ZIP should contain the mapping JSON and all referenced image tiles.",
    )
    if upload is not None:
        signature = (upload.name, len(upload.getvalue()))
        if st.session_state.get("upload_signature") != signature:
            try:
                data, files, json_name = load_bundle(upload)
                initialize_state(data, files, json_name)
                st.session_state.upload_signature = signature
                st.success("Project loaded")
            except Exception as exc:
                st.error(f"Could not load project: {exc}")

    if "data" in st.session_state:
        st.divider()
        st.subheader("Display")
        for i in range(2):
            cfg = st.session_state.settings[i]
            st.caption(f"Channel {i+1}")
            cfg["enabled"] = st.checkbox("Visible", value=cfg["enabled"], key=f"enabled_{i}")
            cfg["lut"] = st.selectbox("LUT", list(LUTS), index=list(LUTS).index(cfg["lut"]), key=f"lut_{i}")
            c1, c2 = st.columns(2)
            cfg["min"] = c1.number_input("Min", value=float(cfg["min"]), format="%.5g", key=f"min_{i}")
            cfg["max"] = c2.number_input("Max", value=float(cfg["max"]), format="%.5g", key=f"max_{i}")

        st.divider()
        if st.button("Clear all anchors", use_container_width=True):
            st.session_state.anchors = []
            st.session_state.matrix = recompute(st.session_state.positions, [])
            st.session_state.pending_click = None
            st.rerun()

if "data" not in st.session_state:
    st.info("Upload the mapping JSON, or preferably a ZIP containing the JSON and its referenced image tiles.")
    st.stop()

try:
    base_image, loaded, missing = build_overview(
        st.session_state.data,
        st.session_state.files,
        st.session_state.settings,
    )
except Exception as exc:
    st.error(f"Could not render the overview: {exc}")
    st.stop()

status_cols = st.columns(4)
status_cols[0].metric("Positions", len(st.session_state.positions))
status_cols[1].metric("Tiles loaded", loaded)
status_cols[2].metric("Tiles missing", missing)
status_cols[3].metric("Anchors", len(st.session_state.anchors))

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.subheader("Overview")
    st.caption("Click a corresponding feature to add an anchor. At least three anchors are required.")
    display_image = draw_overlay(
        base_image,
        st.session_state.positions,
        st.session_state.stage_to_mosaic,
        st.session_state.anchors,
    )
    max_width = min(display_image.width, 1000)
    click = streamlit_image_coordinates(display_image, width=max_width, key=f"overview_{len(st.session_state.anchors)}")
    if click and st.session_state.mosaic_to_stage is not None:
        scale = display_image.width / max_width
        xy = (float(click["x"] * scale), float(click["y"] * scale))
        if st.session_state.get("pending_click") != xy:
            st.session_state.pending_click = xy

    if st.session_state.mosaic_to_stage is None:
        st.warning("This JSON has no usable `affine_A`; image clicks cannot be converted to confocal stage coordinates.")

    if st.session_state.pending_click is not None and st.session_state.mosaic_to_stage is not None:
        xy = st.session_state.pending_click
        stage = apply_affine(st.session_state.mosaic_to_stage, xy)
        with st.form("anchor_form"):
            st.markdown(f"**New anchor** — confocal stage: `{stage[0]:.9f}, {stage[1]:.9f}`")
            c1, c2 = st.columns(2)
            nx = c1.number_input("NanoSIMS X", format="%.6f")
            ny = c2.number_input("NanoSIMS Y", format="%.6f")
            add, cancel = st.columns(2)
            submitted = add.form_submit_button("Add anchor", use_container_width=True)
            cancelled = cancel.form_submit_button("Cancel", use_container_width=True)
            if submitted:
                st.session_state.anchors.append({
                    "mosaic": xy,
                    "stage": stage,
                    "nanosims": (float(nx), float(ny)),
                })
                st.session_state.matrix = recompute(st.session_state.positions, st.session_state.anchors)
                st.session_state.pending_click = None
                st.rerun()
            if cancelled:
                st.session_state.pending_click = None
                st.rerun()

with right:
    tabs = st.tabs(["Coordinates", "Anchors", "Transform"])
    with tabs[0]:
        st.dataframe(
            positions_df(st.session_state.positions),
            use_container_width=True,
            height=430,
            hide_index=True,
            column_config={
                "Confocal X (m)": st.column_config.NumberColumn(format="%.9f"),
                "Confocal Y (m)": st.column_config.NumberColumn(format="%.9f"),
                "Z (m)": st.column_config.NumberColumn(format="%.9f"),
                "NanoSIMS X": st.column_config.NumberColumn(format="%.6f"),
                "NanoSIMS Y": st.column_config.NumberColumn(format="%.6f"),
            },
        )
    with tabs[1]:
        if not st.session_state.anchors:
            st.info("No anchors added.")
        for i, anchor in enumerate(st.session_state.anchors):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    f"**A{i+1}**  \n"
                    f"Confocal: `{anchor['stage'][0]:.9f}, {anchor['stage'][1]:.9f}`  \n"
                    f"NanoSIMS: `{anchor['nanosims'][0]:.6f}, {anchor['nanosims'][1]:.6f}`"
                )
                if c2.button("Delete", key=f"delete_anchor_{i}"):
                    del st.session_state.anchors[i]
                    st.session_state.matrix = recompute(st.session_state.positions, st.session_state.anchors)
                    st.rerun()
    with tabs[2]:
        if st.session_state.matrix is None:
            st.info(f"Add {max(0, 3-len(st.session_state.anchors))} more anchor(s) to calculate the affine transform.")
        else:
            st.code(np.array2string(st.session_state.matrix, precision=10), language="text")
            st.success("NanoSIMS coordinates have been calculated.")

st.divider()
output_name = f"{Path(st.session_state.json_name).stem}_nanosims.json"
st.download_button(
    "Download converted JSON",
    data=export_json(
        st.session_state.data,
        st.session_state.positions,
        st.session_state.anchors,
        st.session_state.matrix,
    ),
    file_name=output_name,
    mime="application/json",
    type="primary",
    disabled=st.session_state.matrix is None,
)
