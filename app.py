from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from streamlit_image_coordinates import streamlit_image_coordinates


LUTS = {
    "Green": (0.0, 1.0, 0.0),
    "Red": (1.0, 0.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "White": (1.0, 1.0, 1.0),
}


# ----------------------------- math helpers -----------------------------
def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least 3 matching source and destination points.")
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
    inv = np.linalg.inv(full)
    return inv[:2, :]


def fmt(v: Optional[float], digits: int = 9) -> str:
    return "" if v is None else f"{v:.{digits}f}"


# ----------------------------- image helpers -----------------------------
def image_to_channel_arrays(img: Image.Image, max_channels: int = 2) -> List[np.ndarray]:
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


def composite_to_rgb(img: Image.Image, channel_settings: List[dict]) -> Image.Image:
    channels = image_to_channel_arrays(img, max_channels=2)
    if not channels:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    h, w = channels[0].shape[:2]
    out = np.zeros((h, w, 3), dtype=float)
    for i, arr in enumerate(channels[:2]):
        settings = channel_settings[i] if i < len(channel_settings) else {}
        if not settings.get("enabled", True):
            continue
        lo = float(settings.get("min", np.nanpercentile(arr, 1)))
        hi = float(settings.get("max", np.nanpercentile(arr, 99.8)))
        lut = LUTS.get(str(settings.get("lut", "Green")), LUTS["Green"])
        norm = np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1)
        for c in range(3):
            out[..., c] += norm * lut[c]
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def load_uploaded_images(files, zip_file) -> Dict[str, bytes]:
    image_bytes: Dict[str, bytes] = {}
    if files:
        for f in files:
            data = f.read()
            # Store both the uploaded path and only the basename. This lets JSONs
            # with Windows paths still match uploaded browser files.
            image_bytes[f.name] = data
            image_bytes[Path(f.name).name] = data
    if zip_file is not None:
        with zipfile.ZipFile(zip_file) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    data = zf.read(name)
                    image_bytes[name] = data
                    image_bytes[Path(name).name] = data
    return image_bytes


def find_tile_bytes(raw_path: str, image_bytes: Dict[str, bytes]) -> Optional[bytes]:
    if raw_path in image_bytes:
        return image_bytes[raw_path]
    return image_bytes.get(Path(raw_path).name)


def default_channel_settings(data: dict, image_bytes: Dict[str, bytes]) -> List[dict]:
    mins: List[List[float]] = [[], []]
    maxs: List[List[float]] = [[], []]
    for tile in (data.get("mosaic") or {}).get("tiles") or []:
        raw = tile.get("path")
        if not raw:
            continue
        b = find_tile_bytes(raw, image_bytes)
        if b is None:
            continue
        try:
            img = Image.open(io.BytesIO(b))
            channels = image_to_channel_arrays(img, max_channels=2)
            for i, arr in enumerate(channels[:2]):
                mins[i].append(float(np.nanpercentile(arr, 1)))
                maxs[i].append(float(np.nanpercentile(arr, 99.8)))
        except Exception:
            continue
    settings = []
    for i, lut in enumerate(["Green", "Red"]):
        lo = min(mins[i]) if mins[i] else 0.0
        hi = max(maxs[i]) if maxs[i] else 1.0
        if hi <= lo:
            hi = lo + 1.0
        settings.append({"enabled": True, "min": lo, "max": hi, "lut": lut})
    return settings


def build_base_overview(
    data: dict,
    image_bytes: Dict[str, bytes],
    channel_settings: List[dict],
    preview_factor: float = 1.0,
) -> Tuple[Image.Image, int, int]:
    """Build the overview image.

    preview_factor < 1 makes a smaller overview for speed. All anchor math still
    uses original mosaic coordinates; display/click conversion accounts for this.
    """
    preview_factor = max(0.05, min(float(preview_factor), 1.0))
    mosaic = data.get("mosaic") or {}
    tiles = mosaic.get("tiles") or []
    size = mosaic.get("size") or [1200, 900]
    try:
        orig_width, orig_height = int(size[0]), int(size[1])
    except Exception:
        orig_width, orig_height = 1200, 900

    width = max(1, int(round(orig_width * preview_factor)))
    height = max(1, int(round(orig_height * preview_factor)))
    overview = Image.new("RGB", (width, height), (18, 18, 18))
    loaded = 0
    missing = 0

    for tile in tiles:
        raw = tile.get("path")
        b = find_tile_bytes(raw, image_bytes) if raw else None
        if b is None:
            missing += 1
            continue
        try:
            img = Image.open(io.BytesIO(b))
            comp = composite_to_rgb(img, channel_settings)
            if preview_factor != 1.0:
                comp = comp.resize(
                    (max(1, int(round(comp.width * preview_factor))), max(1, int(round(comp.height * preview_factor)))),
                    Image.Resampling.BILINEAR,
                )
            ox = int(round(int(tile.get("offset_x", tile.get("ox", 0))) * preview_factor))
            oy = int(round(int(tile.get("offset_y", tile.get("oy", 0))) * preview_factor))
            overview.paste(comp, (ox, oy))
            loaded += 1
        except Exception:
            missing += 1

    if loaded == 0:
        draw = ImageDraw.Draw(overview)
        draw.text((24, 24), "No image tiles found. Upload tiles or a ZIP containing them.", fill=(220, 220, 220))
    return overview, loaded, missing


def safe_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_display_overview(
    base: Image.Image,
    scale: float,
    coord_scale: float,
    anchors: List[dict],
    pending_anchor: Optional[dict],
    positions: Dict[str, dict],
    stage_to_mosaic_A: Optional[np.ndarray],
    show_positions: bool,
    show_anchor_labels: bool,
) -> Image.Image:
    """Resize overview for display, then draw overlays at display scale.

    Drawing after resizing makes anchors visible even when zoomed far out.
    Click coordinates from streamlit-image-coordinates are then converted back
    to original mosaic pixels by dividing by the same scale.
    """
    scale = max(0.05, min(float(scale), 4.0))
    w = max(1, int(round(base.width * scale)))
    h = max(1, int(round(base.height * scale)))
    display = base.resize((w, h), Image.Resampling.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(display)

    marker_r = max(6, int(round(8 * scale)))
    line_w = max(2, int(round(3 * scale)))
    font = safe_font(max(12, int(round(14 * scale))))

    if show_positions and stage_to_mosaic_A is not None:
        pos_r = max(3, int(round(5 * scale)))
        for pid, pos in positions.items():
            stage = pos.get("stage")
            if stage is None:
                continue
            x0, y0 = apply_affine(stage_to_mosaic_A, stage)
            x, y = x0 * coord_scale * scale, y0 * coord_scale * scale
            draw.ellipse((x - pos_r, y - pos_r, x + pos_r, y + pos_r), outline=(80, 213, 255), width=max(1, line_w - 1))
            if scale >= 0.35:
                draw.text((x + pos_r + 3, y - pos_r - 6), str(pid), fill=(232, 251, 255), font=font)

    # Existing anchors: bright filled circles + cross + label.
    for i, anchor in enumerate(anchors, start=1):
        x0, y0 = anchor["mosaic"]
        x, y = x0 * coord_scale * scale, y0 * coord_scale * scale
        draw.ellipse((x - marker_r, y - marker_r, x + marker_r, y + marker_r), fill=(255, 207, 74), outline=(0, 0, 0), width=line_w)
        draw.line((x - marker_r * 1.8, y, x + marker_r * 1.8, y), fill=(255, 255, 255), width=line_w)
        draw.line((x, y - marker_r * 1.8, x, y + marker_r * 1.8), fill=(255, 255, 255), width=line_w)
        if show_anchor_labels:
            draw.text((x + marker_r + 6, y + marker_r + 2), f"A{i}", fill=(255, 255, 255), font=font)

    # Pending anchor: magenta ring so it is obvious before clicking Add anchor.
    if pending_anchor is not None:
        x0, y0 = pending_anchor["mosaic"]
        x, y = x0 * coord_scale * scale, y0 * coord_scale * scale
        r = marker_r + 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 255), width=max(3, line_w))
        draw.line((x - r * 1.6, y, x + r * 1.6, y), fill=(255, 0, 255), width=max(2, line_w - 1))
        draw.line((x, y - r * 1.6, x, y + r * 1.6), fill=(255, 0, 255), width=max(2, line_w - 1))
        draw.text((x + r + 6, y - r), "pending", fill=(255, 0, 255), font=font)

    return display


# ----------------------------- JSON helpers -----------------------------
def parse_positions(data: dict) -> Dict[str, dict]:
    positions: Dict[str, dict] = {}
    entries = (data.get("positions") or {}).get("entries") or {}
    for pid, entry in entries.items():
        coarse = entry.get("coarse_off") or {}
        try:
            stage = (float(coarse["x"]), float(coarse["y"]))
            z = float(coarse.get("z", 0.0))
        except Exception:
            stage = None
            z = None
        positions[str(pid)] = {"entry": entry, "stage": stage, "z": z, "nano": None}
    return positions


def load_affines(data: dict):
    stage_to_mosaic_A = None
    mosaic_to_stage_A = None
    a = data.get("affine_A")
    if a is not None:
        arr = np.array(a, dtype=float)
        if arr.shape == (2, 3):
            stage_to_mosaic_A = arr
            mosaic_to_stage_A = invert_affine(arr)
    return stage_to_mosaic_A, mosaic_to_stage_A


def compute_nanosims(anchors: List[dict], positions: Dict[str, dict]):
    if len(anchors) < 3:
        for pos in positions.values():
            pos["nano"] = None
        return None, positions
    src = np.array([a["stage"] for a in anchors], dtype=float)
    dst = np.array([a["nanosims"] for a in anchors], dtype=float)
    a = affine_from_points(src, dst)
    for pos in positions.values():
        stage = pos.get("stage")
        pos["nano"] = apply_affine(a, stage) if stage is not None else None
    return a, positions


def make_output_json(data: dict, anchors: List[dict], positions: Dict[str, dict], stage_to_nanosims_A: Optional[np.ndarray]) -> str:
    out = copy.deepcopy(data)
    out["nanosims_conversion"] = {
        "anchors": [
            {
                "mosaic": [float(a["mosaic"][0]), float(a["mosaic"][1])],
                "confocal_stage": [float(a["stage"][0]), float(a["stage"][1])],
                "nanosims": [float(a["nanosims"][0]), float(a["nanosims"][1])],
            }
            for a in anchors
        ],
        "stage_to_nanosims_A": stage_to_nanosims_A.tolist() if stage_to_nanosims_A is not None else None,
    }
    entries = ((out.get("positions") or {}).get("entries") or {})
    for pid, pos in positions.items():
        if pid not in entries:
            continue
        nano = pos.get("nano")
        if nano is None:
            entries[pid].pop("nanosims_off", None)
        else:
            entries[pid]["nanosims_off"] = {"x": float(nano[0]), "y": float(nano[1])}
    return json.dumps(out, indent=2)



# ----------------------------- cache/state helpers -----------------------------
def uploaded_signature(json_bytes: bytes, tile_files, zip_file) -> str:
    h = hashlib.sha256()
    h.update(json_bytes)
    if tile_files:
        for f in tile_files:
            h.update(f.name.encode("utf-8", "ignore"))
            h.update(str(getattr(f, "size", 0)).encode())
    if zip_file is not None:
        h.update(zip_file.name.encode("utf-8", "ignore"))
        h.update(str(getattr(zip_file, "size", 0)).encode())
    return h.hexdigest()


def settings_tuple(settings: List[dict]) -> Tuple[Tuple[bool, float, float, str], ...]:
    return tuple((bool(s.get("enabled", True)), float(s.get("min", 0.0)), float(s.get("max", 1.0)), str(s.get("lut", "Green"))) for s in settings)


@st.cache_data(show_spinner=False)
def cached_default_channel_settings(data_json: str, image_items: Tuple[Tuple[str, bytes], ...]) -> List[dict]:
    return default_channel_settings(json.loads(data_json), dict(image_items))


@st.cache_data(show_spinner="Building overview image...")
def cached_build_overview_png(
    data_json: str,
    image_items: Tuple[Tuple[str, bytes], ...],
    settings_items: Tuple[Tuple[bool, float, float, str], ...],
    preview_factor: float,
) -> Tuple[bytes, int, int, int, int]:
    settings = [
        {"enabled": enabled, "min": lo, "max": hi, "lut": lut}
        for enabled, lo, hi, lut in settings_items
    ]
    overview, loaded, missing = build_base_overview(json.loads(data_json), dict(image_items), settings, preview_factor)
    buf = io.BytesIO()
    overview.save(buf, format="PNG")
    return buf.getvalue(), loaded, missing, overview.width, overview.height

# ----------------------------- app -----------------------------
st.set_page_config(page_title="NanoSIMS Converter", layout="wide")
st.title("Overview to NanoSIMS Converter")
st.caption("Streamlit/browser version with stable anchor clicks, cached previews, visible anchors, and zoom controls")

if "anchors" not in st.session_state:
    st.session_state.anchors = []
if "pending_anchor" not in st.session_state:
    st.session_state.pending_anchor = None
if "channel_settings" not in st.session_state:
    st.session_state.channel_settings = None
if "loaded_signature" not in st.session_state:
    st.session_state.loaded_signature = None
if "last_click_xy" not in st.session_state:
    st.session_state.last_click_xy = None

with st.sidebar:
    st.header("1. Upload files")
    json_file = st.file_uploader("Mapping JSON", type=["json"])
    tile_files = st.file_uploader(
        "Image tiles",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
    )
    zip_file = st.file_uploader("Or ZIP containing image tiles", type=["zip"])

    st.header("2. Overview display")
    preview_percent = st.select_slider(
        "Preview resolution",
        options=[25, 50, 75, 100],
        value=50,
        help="Lower values are faster. Anchor math still uses original coordinates.",
    )
    zoom_percent = st.slider("Display zoom", min_value=10, max_value=250, value=100, step=5, help="Lower values zoom out; higher values zoom in.")
    show_positions = st.checkbox("Show sample positions", value=True)
    show_anchor_labels = st.checkbox("Show anchor labels", value=True)

    st.header("3. Anchor controls")
    if st.button("Clear anchors"):
        st.session_state.anchors = []
        st.session_state.pending_anchor = None
        st.rerun()
    if st.button("Clear pending click"):
        st.session_state.pending_anchor = None
        st.session_state.last_click_xy = None
        st.rerun()

if json_file is None:
    st.info("Upload your mapping JSON to start.")
    st.stop()

try:
    data = json.loads(json_file.getvalue().decode("utf-8"))
except Exception as exc:
    st.error(f"Could not read JSON: {exc}")
    st.stop()

image_bytes = load_uploaded_images(tile_files, zip_file)
image_items = tuple(sorted(image_bytes.items(), key=lambda kv: kv[0]))
data_json = json.dumps(data, sort_keys=True)
current_signature = uploaded_signature(json_file.getvalue(), tile_files, zip_file)

# Reset only when the input dataset changes, not when color/zoom widgets rerun.
if st.session_state.loaded_signature != current_signature:
    st.session_state.loaded_signature = current_signature
    st.session_state.anchors = []
    st.session_state.pending_anchor = None
    st.session_state.channel_settings = cached_default_channel_settings(data_json, image_items)

positions = parse_positions(data)
try:
    stage_to_mosaic_A, mosaic_to_stage_A = load_affines(data)
except Exception as exc:
    st.error(f"Could not read affine_A transform: {exc}")
    st.stop()

if mosaic_to_stage_A is None:
    st.error("This JSON does not contain a usable affine_A transform, so anchor clicks cannot be converted to confocal coordinates.")
    st.stop()

if st.session_state.channel_settings is None:
    st.session_state.channel_settings = cached_default_channel_settings(data_json, image_items)

with st.sidebar:
    st.header("4. Colors")
    st.caption("Color changes are applied only when you press Apply, so the overview does not rebuild after every small edit.")
    with st.form("color_form"):
        edited_settings = []
        for i, default in enumerate(st.session_state.channel_settings, start=1):
            st.markdown(f"**Channel {i}**")
            enabled = st.checkbox("Show", value=bool(default["enabled"]), key=f"form_ch{i}_enabled")
            lo = st.number_input("Min", value=float(default["min"]), key=f"form_ch{i}_min", format="%.6f")
            hi = st.number_input("Max", value=float(default["max"]), key=f"form_ch{i}_max", format="%.6f")
            lut_values = list(LUTS.keys())
            lut_index = lut_values.index(default.get("lut", "Green")) if default.get("lut", "Green") in lut_values else 0
            lut = st.selectbox("LUT", lut_values, index=lut_index, key=f"form_ch{i}_lut")
            edited_settings.append({"enabled": enabled, "min": lo, "max": hi, "lut": lut})
        apply_colors = st.form_submit_button("Apply color settings")
    if apply_colors:
        st.session_state.channel_settings = edited_settings
        st.session_state.pending_anchor = None
        st.rerun()
    if st.button("Reset color ranges"):
        st.session_state.channel_settings = cached_default_channel_settings(data_json, image_items)
        st.rerun()

channel_settings = st.session_state.channel_settings

stage_to_nanosims_A, positions = compute_nanosims(st.session_state.anchors, positions)
preview_factor = preview_percent / 100.0
overview_png, loaded, missing, preview_width, preview_height = cached_build_overview_png(
    data_json, image_items, settings_tuple(channel_settings), preview_factor
)
base_overview = Image.open(io.BytesIO(overview_png)).convert("RGB")
scale = zoom_percent / 100.0
shown_overview = make_display_overview(
    base=base_overview,
    scale=scale,
    coord_scale=preview_factor,
    anchors=st.session_state.anchors,
    pending_anchor=st.session_state.pending_anchor,
    positions=positions,
    stage_to_mosaic_A=stage_to_mosaic_A,
    show_positions=show_positions,
    show_anchor_labels=show_anchor_labels,
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Overview")
    st.write(
        f"Tiles loaded: **{loaded}**; missing: **{missing}**. "
        f"Preview image: **{base_overview.width} × {base_overview.height} px** "
        f"({preview_percent}% resolution). Display zoom: **{zoom_percent}%**."
    )
    st.caption("Click the displayed overview to choose an anchor. After clicking, enter NanoSIMS X/Y in the panel on the right, then press Add anchor.")
    click = streamlit_image_coordinates(shown_overview, key="overview_click_stable")
    if click is not None:
        # Convert from displayed-image pixels back to original full-resolution mosaic pixels.
        # Do NOT call st.rerun() here. Some Streamlit image-click components return
        # the last click again after every rerun, which can cause a refresh loop and
        # prevent the NanoSIMS coordinate fields from appearing.
        click_xy = (int(click["x"]), int(click["y"]))
        display_to_original = max(preview_factor * scale, 1e-12)
        uv = (float(click["x"]) / display_to_original, float(click["y"]) / display_to_original)
        stage = apply_affine(mosaic_to_stage_A, uv)
        st.session_state.pending_anchor = {"mosaic": uv, "stage": stage}
        st.session_state.last_click_xy = click_xy
        st.success(f"Anchor point selected at mosaic x={uv[0]:.2f}, y={uv[1]:.2f}. Now enter NanoSIMS X/Y on the right.")

with right:
    st.subheader("Add NanoSIMS anchor")
    pending = st.session_state.get("pending_anchor")
    if pending is None:
        st.write("Click a matching feature/corner on the overview image.")
    else:
        st.write(f"Clicked mosaic: x={pending['mosaic'][0]:.2f}, y={pending['mosaic'][1]:.2f}")
        st.write(f"Confocal stage: x={pending['stage'][0]:.9f}, y={pending['stage'][1]:.9f}")
        nx = st.number_input("NanoSIMS X", value=0.0, format="%.6f", key="pending_nanosims_x")
        ny = st.number_input("NanoSIMS Y", value=0.0, format="%.6f", key="pending_nanosims_y")
        if st.button("Add anchor", type="primary"):
            st.session_state.anchors.append(
                {"mosaic": pending["mosaic"], "stage": pending["stage"], "nanosims": (float(nx), float(ny))}
            )
            st.session_state.pending_anchor = None
            st.session_state.last_click_xy = None
            st.rerun()

    st.subheader("Anchors")
    if st.session_state.anchors:
        anchor_df = pd.DataFrame(
            [
                {
                    "Anchor": f"A{i}",
                    "Mosaic X": fmt(a["mosaic"][0], 2),
                    "Mosaic Y": fmt(a["mosaic"][1], 2),
                    "Confocal X (m)": fmt(a["stage"][0]),
                    "Confocal Y (m)": fmt(a["stage"][1]),
                    "NanoSIMS X": fmt(a["nanosims"][0], 6),
                    "NanoSIMS Y": fmt(a["nanosims"][1], 6),
                }
                for i, a in enumerate(st.session_state.anchors, start=1)
            ]
        )
        st.dataframe(anchor_df, use_container_width=True, hide_index=True)
        remove = st.number_input("Delete anchor number", min_value=1, max_value=len(st.session_state.anchors), value=1, step=1)
        if st.button("Delete selected anchor"):
            del st.session_state.anchors[int(remove) - 1]
            st.rerun()
    else:
        st.write("No anchors yet. Add at least 3 anchors.")

st.subheader("Sample coordinates")
rows = []
for pid in sorted(positions):
    pos = positions[pid]
    stage = pos.get("stage")
    nano = pos.get("nano")
    rows.append(
        {
            "ID": pid,
            "Confocal X (m)": fmt(stage[0] if stage else None),
            "Confocal Y (m)": fmt(stage[1] if stage else None),
            "Z (m)": fmt(pos.get("z")),
            "NanoSIMS X": fmt(nano[0] if nano else None, 6),
            "NanoSIMS Y": fmt(nano[1] if nano else None, 6),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if len(st.session_state.anchors) < 3:
    st.warning(f"{len(st.session_state.anchors)} anchor(s). Add at least 3 to calculate NanoSIMS coordinates.")
else:
    st.success(f"NanoSIMS coordinates calculated from {len(st.session_state.anchors)} anchors.")
    output_json = make_output_json(data, st.session_state.anchors, positions, stage_to_nanosims_A)
    output_name = Path(json_file.name).stem + "_nanosims.json"
    st.download_button("Download converted JSON", output_json, file_name=output_name, mime="application/json")
