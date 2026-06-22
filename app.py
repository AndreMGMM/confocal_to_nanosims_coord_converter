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

st.set_page_config(page_title="Overview to NanoSIMS", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --app-bg:#22314d; --sidebar-bg:#293b5c; --panel:#304565; --panel-2:#385071;
        --input-bg:#314867; --hover-bg:#426085; --border:#587096; --text:#f4f7ff;
        --text-soft:#c6d2e5; --accent:#72aaff;
    }
    html, body, [class*="css"] { color:var(--text); }
    .stApp { color:var(--text); background:radial-gradient(circle at top left,rgba(114,170,255,.16),transparent 38rem),var(--app-bg); }
    [data-testid="stSidebar"] { background:var(--sidebar-bg); border-right:1px solid var(--border); }
    [data-testid="stHeader"] { background:rgba(34,49,77,.94); height:3.4rem; }
    .block-container { max-width:1900px; padding-top:4.4rem !important; padding-bottom:2rem; }
    .app-title { font-size:1.8rem; font-weight:760; letter-spacing:-.03em; margin-bottom:.15rem; }
    .app-subtitle { color:var(--text-soft); margin-bottom:1rem; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input,
    div[data-testid="stFileUploaderDropzone"], textarea {
        background:var(--input-bg)!important; color:var(--text)!important; border-color:var(--border)!important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
        background:var(--panel-2)!important; color:var(--text)!important;
    }
    li[role="option"]:hover, li[aria-selected="true"] { background:var(--hover-bg)!important; }
    div[data-testid="stDataFrame"] { background:var(--panel)!important; border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    div[data-testid="stDataFrame"] canvas { background:var(--panel)!important; }
    [data-testid="stMetric"], [data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(48,69,101,.62); border-color:var(--border)!important; border-radius:10px;
    }
    button[data-baseweb="tab"] { color:var(--text-soft); }
    button[data-baseweb="tab"][aria-selected="true"] { color:var(--text); }
    .stButton>button,.stDownloadButton>button { border-radius:8px; border:1px solid var(--border); background:var(--panel-2); color:var(--text); }
    .stButton>button:hover,.stDownloadButton>button:hover { border-color:var(--accent); background:var(--hover-bg); color:white; }
    .anchor-mode { padding:.65rem .85rem; border:1px solid #d9b632; background:rgba(217,182,50,.14); border-radius:9px; color:#ffe887; }
    </style>
    """,
    unsafe_allow_html=True,
)

LUTS = {
    "Green": (0.0, 1.0, 0.0), "Red": (1.0, 0.0, 0.0), "Blue": (0.0, 0.0, 1.0),
    "White": (1.0, 1.0, 1.0), "Magenta": (1.0, 0.0, 1.0), "Cyan": (0.0, 1.0, 1.0), "Yellow": (1.0, 1.0, 0.0),
}


def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 3 or src.shape[1] != 2:
        raise ValueError("At least three matching 2D source and destination points are required.")
    n = src.shape[0]
    x = np.zeros((2*n, 6), dtype=float); b = np.zeros((2*n,), dtype=float)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
        x[2*i, :3] = [sx, sy, 1.0]; x[2*i+1, 3:] = [sx, sy, 1.0]
        b[2*i] = dx; b[2*i+1] = dy
    a, *_ = np.linalg.lstsq(x, b, rcond=None)
    return np.array([[a[0], a[1], a[2]], [a[3], a[4], a[5]]], dtype=float)


def apply_affine(a: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y = xy
    return float(a[0,0]*x+a[0,1]*y+a[0,2]), float(a[1,0]*x+a[1,1]*y+a[1,2])


def invert_affine(a: np.ndarray) -> np.ndarray:
    full = np.array([[a[0,0],a[0,1],a[0,2]],[a[1,0],a[1,1],a[1,2]],[0.,0.,1.]])
    return np.linalg.inv(full)[:2,:]


def normalized_name(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1].strip().lower()


def add_file_aliases(files: Dict[str, bytes], name: str, raw: bytes) -> None:
    clean = name.replace("\\", "/").strip().lower(); base = normalized_name(clean)
    files[clean] = raw; files[base] = raw; files[Path(base).stem.lower()] = raw


def resolve_uploaded_file(files: Dict[str, bytes], raw_path: str) -> Optional[bytes]:
    clean = str(raw_path or "").replace("\\", "/").strip().lower(); base = normalized_name(clean)
    for key in (clean, base, Path(base).stem.lower()):
        if key in files: return files[key]
    stem = Path(base).stem.lower()
    for key, value in files.items():
        key_base = normalized_name(key)
        if key_base == base or Path(key_base).stem.lower() == stem: return value
    return None


def load_bundle(uploaded_files) -> Tuple[dict, Dict[str, bytes], str]:
    files: Dict[str, bytes] = {}; json_candidates: List[str] = []
    for uploaded in uploaded_files:
        if uploaded.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    raw = zf.read(info.filename); add_file_aliases(files, info.filename, raw)
                    if info.filename.lower().endswith(".json"): json_candidates.append(normalized_name(info.filename))
        else:
            raw = uploaded.getvalue(); add_file_aliases(files, uploaded.name, raw)
            if uploaded.name.lower().endswith(".json"): json_candidates.append(normalized_name(uploaded.name))
    json_candidates = list(dict.fromkeys(json_candidates))
    if not json_candidates: raise ValueError("Select a mapping JSON, or a ZIP containing one.")
    json_name = json_candidates[0]
    return json.loads(files[json_name].decode("utf-8-sig")), files, json_name


def image_to_channel_arrays(raw: bytes, max_channels: int = 2) -> List[np.ndarray]:
    img = Image.open(io.BytesIO(raw)); channels: List[np.ndarray] = []
    for frame in ImageSequence.Iterator(img):
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            for c in range(min(arr.shape[-1], max_channels-len(channels))):
                channels.append(arr[...,c].astype(np.float32))
                if len(channels) >= max_channels: return channels
        else:
            channels.append(arr.astype(np.float32))
            if len(channels) >= max_channels: return channels
    return channels


def prepare_tiles(data: dict, files: Dict[str, bytes]):
    prepared = []; mins=[[],[]]; maxs=[[],[]]; missing=[]
    for tile in (data.get("mosaic") or {}).get("tiles") or []:
        path = str(tile.get("path") or ""); raw = resolve_uploaded_file(files, path)
        if raw is None: missing.append(normalized_name(path)); continue
        try:
            ch = image_to_channel_arrays(raw, 2)
            for i, arr in enumerate(ch[:2]):
                mins[i].append(float(np.nanpercentile(arr,1))); maxs[i].append(float(np.nanpercentile(arr,99.8)))
            prepared.append((int(tile.get("offset_x",tile.get("ox",0))), int(tile.get("offset_y",tile.get("oy",0))), ch))
        except Exception:
            missing.append(normalized_name(path))
    settings=[]
    for i,lut in enumerate(("Green","Red")):
        lo=min(mins[i]) if mins[i] else 0.; hi=max(maxs[i]) if maxs[i] else 1.
        settings.append({"enabled":True,"min":lo,"max":max(hi,lo+1.),"lut":lut})
    return prepared, settings, missing


def render_mosaic(data: dict, prepared_tiles, settings: List[dict]) -> Image.Image:
    size=(data.get("mosaic") or {}).get("size") or [1200,900]
    overview=Image.new("RGB",(max(int(size[0]),1),max(int(size[1]),1)),(28,38,57))
    for ox,oy,channels in prepared_tiles:
        h,w=channels[0].shape[:2]; out=np.zeros((h,w,3),dtype=np.float32)
        for i,arr in enumerate(channels[:2]):
            cfg=settings[i]
            if not cfg["enabled"]: continue
            lo,hi=float(cfg["min"]),float(cfg["max"]); norm=np.clip((arr-lo)/max(hi-lo,1e-12),0,1)
            lut=LUTS[cfg["lut"]]
            out[...,0]+=norm*lut[0]; out[...,1]+=norm*lut[1]; out[...,2]+=norm*lut[2]
        overview.paste(Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8),"RGB"),(ox,oy))
    return overview


def get_affines(data: dict):
    raw=data.get("affine_A")
    if raw is None: return None,None
    arr=np.asarray(raw,dtype=float)
    if arr.shape!=(2,3): return None,None
    return arr,invert_affine(arr)


def get_positions(data: dict) -> Dict[str,dict]:
    positions={}
    for pid,entry in ((data.get("positions") or {}).get("entries") or {}).items():
        coarse=entry.get("coarse_off") or {}
        try: stage=(float(coarse["x"]),float(coarse["y"])); z=float(coarse.get("z",0.))
        except Exception: stage=None; z=None
        positions[str(pid)]={"stage":stage,"z":z,"nano":None}
    return positions


def recompute(positions,anchors):
    if len(anchors)<3:
        for pos in positions.values(): pos["nano"]=None
        return None
    matrix=affine_from_points(np.asarray([a["stage"] for a in anchors]),np.asarray([a["nanosims"] for a in anchors]))
    for pos in positions.values(): pos["nano"]=apply_affine(matrix,pos["stage"]) if pos["stage"] is not None else None
    return matrix


def positions_df(positions):
    rows=[]
    for pid in sorted(positions):
        p=positions[pid]; stage=p["stage"]; nano=p["nano"]
        rows.append({"ID":pid,"Confocal X (m)":stage[0] if stage else np.nan,"Confocal Y (m)":stage[1] if stage else np.nan,
                     "Z (m)":p["z"] if p["z"] is not None else np.nan,"NanoSIMS X":nano[0] if nano else np.nan,"NanoSIMS Y":nano[1] if nano else np.nan})
    return pd.DataFrame(rows)


def export_json(data,positions,anchors,matrix):
    result=copy.deepcopy(data)
    result["nanosims_conversion"]={"anchors":[{"mosaic":list(map(float,a["mosaic"])),"confocal_stage":list(map(float,a["stage"])),"nanosims":list(map(float,a["nanosims"]))} for a in anchors],"stage_to_nanosims_A":matrix.tolist() if matrix is not None else None}
    entries=((result.get("positions") or {}).get("entries") or {})
    for pid,pos in positions.items():
        if pid not in entries: continue
        if pos["nano"] is None: entries[pid].pop("nanosims_off",None)
        else: entries[pid]["nanosims_off"]={"x":float(pos["nano"][0]),"y":float(pos["nano"][1])}
    return json.dumps(result,indent=2).encode("utf-8")


def draw_overlays(image: Image.Image, positions, stage_to_mosaic, anchors) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if stage_to_mosaic is not None:
        for pid, p in positions.items():
            if p["stage"] is None:
                continue
            x, y = apply_affine(stage_to_mosaic, p["stage"])
            draw.ellipse((x-4, y-4, x+4, y+4), outline=(103, 216, 255), width=2)
            draw.text((x+7, y-12), str(pid), fill=(225, 247, 255))
    for i, anchor in enumerate(anchors, 1):
        x, y = anchor["mosaic"]
        arm = 15
        draw.line((x-arm, y, x+arm, y), fill=(255, 216, 45), width=4)
        draw.line((x, y-arm, x, y+arm), fill=(255, 216, 45), width=4)
        draw.ellipse((x-5, y-5, x+5, y+5), outline=(255, 248, 175), width=2)
        draw.rectangle((x+12, y-21, x+48, y+2), fill=(42, 51, 68), outline=(255, 216, 45), width=2)
        draw.text((x+18, y-18), f"A{i}", fill=(255, 235, 111))
    return out


def make_viewport(image: Image.Image, zoom: float, centre_x: float, centre_y: float, display_width: int = 980):
    zoom = max(1.0, float(zoom))
    crop_w = max(1, int(image.width / zoom))
    crop_h = max(1, int(image.height / zoom))
    cx = int(np.clip(centre_x, crop_w / 2, image.width - crop_w / 2)) if crop_w < image.width else image.width // 2
    cy = int(np.clip(centre_y, crop_h / 2, image.height - crop_h / 2)) if crop_h < image.height else image.height // 2
    left = max(0, min(image.width - crop_w, cx - crop_w // 2))
    top = max(0, min(image.height - crop_h, cy - crop_h // 2))
    crop = image.crop((left, top, left + crop_w, top + crop_h))
    shown_w = min(display_width, max(1, crop.width))
    scale = crop.width / shown_w
    shown_h = max(1, int(crop.height / scale))
    shown = crop.resize((shown_w, shown_h), Image.Resampling.LANCZOS)
    return shown, (left, top), scale

def init_state(data,files,json_name):
    prepared,settings,missing=prepare_tiles(data,files)
    st.session_state.data=data; st.session_state.files=files; st.session_state.json_name=json_name
    st.session_state.prepared_tiles=prepared; st.session_state.settings=settings; st.session_state.applied_settings=copy.deepcopy(settings)
    st.session_state.missing_tiles=missing; st.session_state.base_image=render_mosaic(data,prepared,settings)
    st.session_state.positions=get_positions(data); st.session_state.stage_to_mosaic,st.session_state.mosaic_to_stage=get_affines(data)
    st.session_state.anchors=[]; st.session_state.matrix=None; st.session_state.pending_click=None; st.session_state.add_anchor_mode=False
    st.session_state.zoom_level=1.0; st.session_state.view_centre_x=data.get("mosaic",{}).get("size",[1200,900])[0]/2; st.session_state.view_centre_y=data.get("mosaic",{}).get("size",[1200,900])[1]/2

st.markdown('<div class="app-title">Overview to NanoSIMS Converter</div>',unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Browser-based mapping, anchor fitting and coordinate export.</div>',unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Project")
    uploads=st.file_uploader("Project files",type=["json","zip","tif","tiff","png","jpg","jpeg"],accept_multiple_files=True,help="Select one ZIP, or the mapping JSON and all image tiles together.")
    if uploads:
        signature=tuple(sorted((u.name,len(u.getvalue())) for u in uploads))
        if st.session_state.get("upload_signature")!=signature:
            try:
                data,files,json_name=load_bundle(uploads); init_state(data,files,json_name); st.session_state.upload_signature=signature; st.success("Project loaded")
            except Exception as exc: st.error(f"Could not load project: {exc}")
    if "data" in st.session_state:
        st.divider(); st.subheader("Display")
        with st.form("display_form"):
            draft=[]
            for i in range(2):
                cfg=st.session_state.settings[i]; st.caption(f"Channel {i+1}")
                enabled=st.checkbox("Visible",value=cfg["enabled"],key=f"enabled_{i}")
                lut=st.selectbox("LUT",list(LUTS),index=list(LUTS).index(cfg["lut"]),key=f"lut_{i}")
                c1,c2=st.columns(2); lo=c1.number_input("Min",value=float(cfg["min"]),format="%.5g",key=f"min_{i}"); hi=c2.number_input("Max",value=float(cfg["max"]),format="%.5g",key=f"max_{i}")
                draft.append({"enabled":enabled,"lut":lut,"min":lo,"max":hi})
            apply_display=st.form_submit_button("Apply display settings",use_container_width=True)
        if apply_display:
            st.session_state.settings=draft; st.session_state.applied_settings=copy.deepcopy(draft)
            st.session_state.base_image=render_mosaic(st.session_state.data,st.session_state.prepared_tiles,draft)
            st.rerun()
        st.caption("Image rendering only runs when this button is pressed.")
        st.divider()
        if st.button("Clear all anchors",use_container_width=True):
            st.session_state.anchors=[]; st.session_state.matrix=recompute(st.session_state.positions,[]); st.session_state.pending_click=None; st.session_state.add_anchor_mode=False; st.rerun()

if "data" not in st.session_state:
    st.info("Upload one ZIP, or select the mapping JSON and all referenced image tiles together."); st.stop()

status=st.columns(4)
status[0].metric("Positions",len(st.session_state.positions)); status[1].metric("Tiles loaded",len(st.session_state.prepared_tiles)); status[2].metric("Tiles missing",len(st.session_state.missing_tiles)); status[3].metric("Anchors",len(st.session_state.anchors))
if not st.session_state.prepared_tiles:
    with st.expander("Why is the overview blank?",expanded=True):
        st.warning("No referenced image tiles were matched.")
        if st.session_state.missing_tiles: st.caption("Expected: "+", ".join(st.session_state.missing_tiles[:8]))

left,right=st.columns([1.55,1],gap="large")
with left:
    st.subheader("Overview")
    view_controls = st.columns([1.2, 1, 1])
    zoom = view_controls[0].slider("Zoom", 1.0, 6.0, float(st.session_state.zoom_level), 0.25, key="zoom_control")
    st.session_state.zoom_level = zoom
    image_w, image_h = st.session_state.base_image.size
    centre_x = view_controls[1].number_input("View centre X", min_value=0.0, max_value=float(image_w), value=float(np.clip(st.session_state.view_centre_x,0,image_w)), step=max(1.0,image_w/100), key="centre_x_control")
    centre_y = view_controls[2].number_input("View centre Y", min_value=0.0, max_value=float(image_h), value=float(np.clip(st.session_state.view_centre_y,0,image_h)), step=max(1.0,image_h/100), key="centre_y_control")
    st.session_state.view_centre_x, st.session_state.view_centre_y = centre_x, centre_y
    overlay = draw_overlays(st.session_state.base_image, st.session_state.positions, st.session_state.stage_to_mosaic, st.session_state.anchors)
    shown, origin, scale = make_viewport(overlay, zoom, centre_x, centre_y)
    click = streamlit_image_coordinates(shown, width=shown.width, key=f"overview_click_{st.session_state.add_anchor_mode}_{len(st.session_state.anchors)}")
    if click and st.session_state.add_anchor_mode and st.session_state.mosaic_to_stage is not None:
        full_xy = (origin[0] + float(click["x"]) * scale, origin[1] + float(click["y"]) * scale)
        st.session_state.pending_click = full_xy
        st.session_state.add_anchor_mode = False
        st.rerun()
    if st.session_state.mosaic_to_stage is None:
        st.warning("This JSON has no usable `affine_A`; clicks cannot be converted to confocal stage coordinates.")

with right:
    tabs=st.tabs(["Coordinates","Anchors","Transform"])
    with tabs[0]:
        df=positions_df(st.session_state.positions)
        styled=df.style.format({"Confocal X (m)":"{:.9f}","Confocal Y (m)":"{:.9f}","Z (m)":"{:.9f}","NanoSIMS X":"{:.6f}","NanoSIMS Y":"{:.6f}"},na_rep="")
        st.dataframe(styled,use_container_width=True,height=430,hide_index=True)
    with tabs[1]:
        if st.button("＋ Add anchor",type="primary",use_container_width=True,disabled=st.session_state.mosaic_to_stage is None):
            st.session_state.add_anchor_mode=True; st.session_state.pending_click=None; st.rerun()
        if st.session_state.add_anchor_mode:
            st.markdown('<div class="anchor-mode">Add-anchor mode is active. Click the required feature in the overview.</div>',unsafe_allow_html=True)
        if st.session_state.pending_click is not None and st.session_state.mosaic_to_stage is not None:
            xy=st.session_state.pending_click; stage=apply_affine(st.session_state.mosaic_to_stage,xy)
            st.markdown(f"**New anchor A{len(st.session_state.anchors)+1}**  \nOverview: `{xy[0]:.2f}, {xy[1]:.2f}`  \nConfocal: `{stage[0]:.9f}, {stage[1]:.9f}`")
            initial=pd.DataFrame([{"NanoSIMS X":0.0,"NanoSIMS Y":0.0}])
            with st.form("anchor_values_form"):
                edited=st.data_editor(initial,use_container_width=True,hide_index=True,num_rows="fixed",column_config={"NanoSIMS X":st.column_config.NumberColumn(format="%.6f"),"NanoSIMS Y":st.column_config.NumberColumn(format="%.6f")})
                a,b=st.columns(2); save=a.form_submit_button("Save anchor",use_container_width=True); cancel=b.form_submit_button("Cancel",use_container_width=True)
            if save:
                nx=float(edited.iloc[0]["NanoSIMS X"]); ny=float(edited.iloc[0]["NanoSIMS Y"])
                st.session_state.anchors.append({"mosaic":xy,"stage":stage,"nanosims":(nx,ny)})
                st.session_state.matrix=recompute(st.session_state.positions,st.session_state.anchors); st.session_state.pending_click=None; st.rerun()
            if cancel: st.session_state.pending_click=None; st.rerun()
        for i,anchor in enumerate(st.session_state.anchors):
            with st.container(border=True):
                c1,c2=st.columns([4,1]); c1.markdown(f"**A{i+1}**  \nConfocal: `{anchor['stage'][0]:.9f}, {anchor['stage'][1]:.9f}`  \nNanoSIMS: `{anchor['nanosims'][0]:.6f}, {anchor['nanosims'][1]:.6f}`")
                if c2.button("Delete",key=f"delete_anchor_{i}"):
                    del st.session_state.anchors[i]; st.session_state.matrix=recompute(st.session_state.positions,st.session_state.anchors); st.rerun()
    with tabs[2]:
        if st.session_state.matrix is None: st.info(f"Add {max(0,3-len(st.session_state.anchors))} more anchor(s) to calculate the affine transform.")
        else: st.code(np.array2string(st.session_state.matrix,precision=10),language="text"); st.success("NanoSIMS coordinates have been calculated.")

st.divider(); output_name=f"{Path(st.session_state.json_name).stem}_nanosims.json"
st.download_button("Download converted JSON",data=export_json(st.session_state.data,st.session_state.positions,st.session_state.anchors,st.session_state.matrix),file_name=output_name,mime="application/json",type="primary",disabled=st.session_state.matrix is None)
