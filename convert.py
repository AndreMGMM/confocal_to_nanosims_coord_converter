#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageSequence, ImageTk


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


def fmt_m(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.9f}"


def fmt_ns(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


LUTS = {
    "Green": (0.0, 1.0, 0.0),
    "Red": (1.0, 0.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "White": (1.0, 1.0, 1.0),
}


def _image_to_channel_arrays(path: Path, max_channels: int = 2) -> List[np.ndarray]:
    img = Image.open(path)
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


def composite_to_rgb(path: Path, channel_settings: List[dict]) -> Image.Image:
    channels = _image_to_channel_arrays(path, max_channels=2)
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


class OverviewCanvas(tk.Canvas):
    def __init__(self, master, app: "NanoSimsConverter", **kwargs):
        super().__init__(master, background="#111111", highlightthickness=0, **kwargs)
        self.app = app
        self.base_img: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag: Optional[Tuple[int, int, float, float]] = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._left_click)
        self.bind("<ButtonPress-3>", self._pan_start)
        self.bind("<B3-Motion>", self._pan_move)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.12))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.12))

    def set_image(self, img: Image.Image) -> None:
        self.base_img = img.convert("RGB")
        self.scale = min(
            max((self.winfo_width() or 900) / max(img.width, 1), 0.02),
            max((self.winfo_height() or 700) / max(img.height, 1), 0.02),
        )
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw()

    def image_to_screen(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        if self.base_img is None:
            return 0.0, 0.0
        ox, oy = self._origin()
        return ox + xy[0] * self.scale, oy + xy[1] * self.scale

    def screen_to_image(self, sx: float, sy: float) -> Optional[Tuple[float, float]]:
        if self.base_img is None:
            return None
        ox, oy = self._origin()
        x = (sx - ox) / self.scale
        y = (sy - oy) / self.scale
        if x < 0 or y < 0 or x >= self.base_img.width or y >= self.base_img.height:
            return None
        return float(x), float(y)

    def _origin(self) -> Tuple[float, float]:
        if self.base_img is None:
            return 0.0, 0.0
        w = self.base_img.width * self.scale
        h = self.base_img.height * self.scale
        return (
            (self.winfo_width() - w) / 2.0 + self.pan_x,
            (self.winfo_height() - h) / 2.0 + self.pan_y,
        )

    def redraw(self) -> None:
        self.delete("all")
        if self.base_img is None:
            self.create_text(
                self.winfo_width() / 2,
                self.winfo_height() / 2,
                text="Load mapping JSON",
                fill="#dddddd",
                font=("Segoe UI", 16),
            )
            return
        size = (
            max(1, int(self.base_img.width * self.scale)),
            max(1, int(self.base_img.height * self.scale)),
        )
        view = self.base_img.resize(size, Image.Resampling.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(view)
        ox, oy = self._origin()
        self.create_image(ox, oy, image=self.tk_img, anchor="nw")
        self._draw_positions()
        self._draw_anchors()

    def _draw_positions(self) -> None:
        for pid, pos in self.app.positions.items():
            stage = pos.get("stage")
            if stage is None:
                continue
            uv = self.app.stage_to_mosaic(stage)
            if uv is None:
                continue
            sx, sy = self.image_to_screen(uv)
            r = 5
            self.create_oval(sx - r, sy - r, sx + r, sy + r, outline="#50d5ff", width=2)
            self.create_text(sx + 8, sy - 8, text=pid, fill="#e8fbff", anchor="sw", font=("Segoe UI", 9, "bold"))

    def _draw_anchors(self) -> None:
        for i, anchor in enumerate(self.app.anchors, start=1):
            uv = anchor["mosaic"]
            sx, sy = self.image_to_screen(uv)
            r = 7
            self.create_line(sx - r, sy, sx + r, sy, fill="#ffcf4a", width=2)
            self.create_line(sx, sy - r, sx, sy + r, fill="#ffcf4a", width=2)
            self.create_text(sx + 10, sy + 8, text=f"A{i}", fill="#ffeb9c", anchor="nw", font=("Segoe UI", 10, "bold"))

    def _left_click(self, event) -> None:
        uv = self.screen_to_image(event.x, event.y)
        if uv is not None:
            self.app.on_canvas_click(uv)

    def _pan_start(self, event) -> None:
        self._drag = (event.x, event.y, self.pan_x, self.pan_y)

    def _pan_move(self, event) -> None:
        if self._drag is None:
            return
        x0, y0, px, py = self._drag
        self.pan_x = px + event.x - x0
        self.pan_y = py + event.y - y0
        self.redraw()

    def _wheel(self, event) -> None:
        self._zoom_at(event.x, event.y, 1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom_at(self, sx: float, sy: float, factor: float) -> None:
        before = self.screen_to_image(sx, sy)
        self.scale = min(max(self.scale * factor, 0.02), 20.0)
        if before is not None and self.base_img is not None:
            ox, oy = self._origin()
            self.pan_x += sx - (ox + before[0] * self.scale)
            self.pan_y += sy - (oy + before[1] * self.scale)
        self.redraw()


class NanoSimsConverter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Overview to NanoSIMS Converter")
        self.geometry("1280x780")
        self.minsize(980, 620)

        self.json_path: Optional[Path] = None
        self.data: dict = {}
        self.positions: Dict[str, dict] = {}
        self.stage_to_mosaic_A: Optional[np.ndarray] = None
        self.mosaic_to_stage_A: Optional[np.ndarray] = None
        self.stage_to_nanosims_A: Optional[np.ndarray] = None
        self.anchors: List[dict] = []
        self.add_anchor_mode = False
        self.channel_settings: List[dict] = [
            {"enabled": True, "min": 0.0, "max": 1.0, "lut": "Green"},
            {"enabled": True, "min": 0.0, "max": 1.0, "lut": "Red"},
        ]
        self.last_tile_counts: Tuple[int, int] = (0, 0)

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=480)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Button(toolbar, text="Load JSON", command=self.on_load_json).pack(side="left")
        ttk.Button(toolbar, text="Adjust Colors", command=self.on_adjust_colors).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Add NanoSIMS Anchor", command=self.on_add_anchor).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Clear Anchors", command=self.on_clear_anchors).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Save Converted JSON", command=self.on_save_json).pack(side="left", padx=(8, 0))
        self.status_var = tk.StringVar(value="Load a mapping JSON to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=14)

        self.canvas = OverviewCanvas(self, self)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        side = ttk.Frame(self, padding=(8, 0, 8, 8))
        side.grid(row=1, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        side.rowconfigure(3, weight=1)
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Sample Coordinates", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(
            side,
            columns=("id", "conf_x", "conf_y", "conf_z", "nano_x", "nano_y"),
            show="headings",
            selectmode="browse",
            height=14,
        )
        headings = {
            "id": "ID",
            "conf_x": "Confocal X (m)",
            "conf_y": "Confocal Y (m)",
            "conf_z": "Z (m)",
            "nano_x": "NanoSIMS X",
            "nano_y": "NanoSIMS Y",
        }
        widths = {"id": 58, "conf_x": 105, "conf_y": 105, "conf_z": 92, "nano_x": 105, "nano_y": 105}
        for col, text in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=widths[col], minwidth=50, anchor="e" if col != "id" else "w")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(4, 10))

        ttk.Label(side, text="NanoSIMS Anchors", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.anchor_tree = ttk.Treeview(
            side,
            columns=("id", "conf_x", "conf_y", "nano_x", "nano_y"),
            show="headings",
            selectmode="browse",
            height=8,
        )
        anchor_headings = {
            "id": "Anchor",
            "conf_x": "Confocal X (m)",
            "conf_y": "Confocal Y (m)",
            "nano_x": "NanoSIMS X",
            "nano_y": "NanoSIMS Y",
        }
        for col, text in anchor_headings.items():
            self.anchor_tree.heading(col, text=text)
            self.anchor_tree.column(col, width=92, minwidth=55, anchor="e" if col != "id" else "w")
        self.anchor_tree.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        ttk.Button(side, text="Delete Selected Anchor", command=self.on_delete_anchor).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    def on_load_json(self) -> None:
        start_dir = r"F:\ImSpector_Test\ImSpector_Test" if os.path.isdir(r"F:\ImSpector_Test\ImSpector_Test") else os.getcwd()
        path = filedialog.askopenfilename(
            title="Load mapping JSON",
            initialdir=start_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_json(Path(path))

    def load_json(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load JSON", f"Could not read JSON:\n{exc}")
            return

        self.json_path = path
        self.anchors.clear()
        self.stage_to_nanosims_A = None
        self._load_affine()
        self._load_positions()
        self._init_channel_settings(path)
        img, loaded, missing = self._build_overview_image(path)
        self.last_tile_counts = (loaded, missing)
        self.canvas.set_image(img)
        self._refresh_trees()
        self.status_var.set(f"Loaded {path.name}: {len(self.positions)} positions, {loaded} tiles loaded, {missing} missing.")

    def _load_affine(self) -> None:
        self.stage_to_mosaic_A = None
        self.mosaic_to_stage_A = None
        a = self.data.get("affine_A")
        if a is None:
            return
        try:
            arr = np.array(a, dtype=float)
            if arr.shape == (2, 3):
                self.stage_to_mosaic_A = arr
                self.mosaic_to_stage_A = invert_affine(arr)
        except Exception:
            self.stage_to_mosaic_A = None
            self.mosaic_to_stage_A = None

    def _load_positions(self) -> None:
        self.positions.clear()
        entries = (self.data.get("positions") or {}).get("entries") or {}
        for pid, entry in entries.items():
            coarse = entry.get("coarse_off") or {}
            try:
                stage = (float(coarse["x"]), float(coarse["y"]))
                z = float(coarse.get("z", 0.0))
            except Exception:
                stage = None
                z = None
            self.positions[str(pid)] = {"entry": entry, "stage": stage, "z": z, "nano": None}

    def _resolve_tile_path(self, raw_path: str, json_path: Path) -> Optional[Path]:
        direct = Path(raw_path)
        if direct.is_file():
            return direct
        name = direct.name
        for base in (json_path.parent, json_path.parent.parent):
            candidate = base / name
            if candidate.is_file():
                return candidate
        try:
            matches = list(json_path.parent.glob(name))
            if matches:
                return matches[0]
        except Exception:
            pass
        return None

    def _iter_resolved_tiles(self, json_path: Path):
        mosaic = self.data.get("mosaic") or {}
        for tile in mosaic.get("tiles") or []:
            raw = tile.get("path")
            if not raw:
                yield tile, None
                continue
            yield tile, self._resolve_tile_path(raw, json_path)

    def _init_channel_settings(self, json_path: Path) -> None:
        mins: List[List[float]] = [[], []]
        maxs: List[List[float]] = [[], []]
        for _tile, tpath in self._iter_resolved_tiles(json_path):
            if tpath is None:
                continue
            try:
                channels = _image_to_channel_arrays(tpath, max_channels=2)
            except Exception:
                continue
            for i, arr in enumerate(channels[:2]):
                mins[i].append(float(np.nanpercentile(arr, 1)))
                maxs[i].append(float(np.nanpercentile(arr, 99.8)))

        defaults = [("Green", True), ("Red", True)]
        new_settings: List[dict] = []
        for i, (lut, enabled) in enumerate(defaults):
            lo = min(mins[i]) if mins[i] else 0.0
            hi = max(maxs[i]) if maxs[i] else 1.0
            if hi <= lo:
                hi = lo + 1.0
            new_settings.append({"enabled": enabled, "min": lo, "max": hi, "lut": lut})
        self.channel_settings = new_settings

    def _build_overview_image(self, json_path: Path) -> Tuple[Image.Image, int, int]:
        mosaic = self.data.get("mosaic") or {}
        tiles = mosaic.get("tiles") or []
        size = mosaic.get("size") or [1200, 900]
        try:
            width, height = int(size[0]), int(size[1])
        except Exception:
            width, height = 1200, 900
        overview = Image.new("RGB", (max(width, 1), max(height, 1)), (18, 18, 18))
        loaded = 0
        missing = 0

        for tile, tpath in self._iter_resolved_tiles(json_path):
            if tpath is None:
                missing += 1
                continue
            try:
                img = composite_to_rgb(tpath, self.channel_settings)
                ox = int(tile.get("offset_x", tile.get("ox", 0)))
                oy = int(tile.get("offset_y", tile.get("oy", 0)))
                overview.paste(img, (ox, oy))
                loaded += 1
            except Exception:
                missing += 1

        if loaded == 0:
            draw = ImageDraw.Draw(overview)
            draw.text((24, 24), "Overview image tiles were not found. Coordinates are still loaded.", fill=(220, 220, 220))
        return overview, loaded, missing

    def stage_to_mosaic(self, stage: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.stage_to_mosaic_A is None:
            return None
        return apply_affine(self.stage_to_mosaic_A, stage)

    def mosaic_to_stage(self, uv: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.mosaic_to_stage_A is None:
            return None
        return apply_affine(self.mosaic_to_stage_A, uv)

    def _rebuild_overview(self) -> None:
        if self.json_path is None:
            return
        img, loaded, missing = self._build_overview_image(self.json_path)
        self.last_tile_counts = (loaded, missing)
        self.canvas.set_image(img)
        self.status_var.set(f"Colors updated: {loaded} tiles loaded, {missing} missing.")

    def on_adjust_colors(self) -> None:
        if self.json_path is None:
            messagebox.showinfo("Adjust Colors", "Load a mapping JSON first.")
            return

        win = tk.Toplevel(self)
        win.title("Adjust Colors")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)

        vars_by_channel = []
        headers = ("Show", "Min", "Max", "LUT")
        for c, header in enumerate(headers):
            ttk.Label(win, text=header).grid(row=0, column=c + 1, padx=6, pady=(10, 4), sticky="w")

        for i in range(2):
            settings = self.channel_settings[i]
            enabled_var = tk.BooleanVar(value=bool(settings.get("enabled", True)))
            min_var = tk.StringVar(value=f"{float(settings.get('min', 0.0)):.6g}")
            max_var = tk.StringVar(value=f"{float(settings.get('max', 1.0)):.6g}")
            lut_var = tk.StringVar(value=str(settings.get("lut", "Green")))
            vars_by_channel.append((enabled_var, min_var, max_var, lut_var))

            ttk.Label(win, text=f"Channel {i + 1}").grid(row=i + 1, column=0, padx=10, pady=5, sticky="w")
            ttk.Checkbutton(win, variable=enabled_var).grid(row=i + 1, column=1, padx=6, pady=5)
            ttk.Entry(win, textvariable=min_var, width=12).grid(row=i + 1, column=2, padx=6, pady=5)
            ttk.Entry(win, textvariable=max_var, width=12).grid(row=i + 1, column=3, padx=6, pady=5)
            ttk.Combobox(win, textvariable=lut_var, values=list(LUTS.keys()), state="readonly", width=9).grid(
                row=i + 1, column=4, padx=6, pady=5
            )

        buttons = ttk.Frame(win)
        buttons.grid(row=3, column=0, columnspan=5, sticky="e", padx=10, pady=(8, 10))

        def apply_and_close() -> None:
            new_settings = []
            try:
                for enabled_var, min_var, max_var, lut_var in vars_by_channel:
                    lo = float(min_var.get())
                    hi = float(max_var.get())
                    if hi <= lo:
                        raise ValueError("Max must be greater than min.")
                    new_settings.append(
                        {
                            "enabled": bool(enabled_var.get()),
                            "min": lo,
                            "max": hi,
                            "lut": lut_var.get() if lut_var.get() in LUTS else "Green",
                        }
                    )
            except Exception as exc:
                messagebox.showwarning("Adjust Colors", f"Check the color ranges:\n{exc}", parent=win)
                return
            self.channel_settings = new_settings
            win.destroy()
            self._rebuild_overview()

        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(buttons, text="OK", command=apply_and_close).pack(side="right", padx=(0, 8))

        self.wait_window(win)

    def on_add_anchor(self) -> None:
        if self.canvas.base_img is None:
            messagebox.showinfo("Anchor", "Load a mapping JSON first.")
            return
        if self.mosaic_to_stage_A is None:
            messagebox.showwarning("Anchor", "This JSON does not contain a usable affine_A transform.")
            return
        self.add_anchor_mode = True
        self.status_var.set("Click the matching corner/feature on the overview, then enter its NanoSIMS coordinates.")

    def on_canvas_click(self, uv: Tuple[float, float]) -> None:
        if not self.add_anchor_mode:
            return
        self.add_anchor_mode = False
        stage = self.mosaic_to_stage(uv)
        if stage is None:
            messagebox.showwarning("Anchor", "Could not convert this image point to confocal coordinates.")
            return
        coords = self._ask_nanosims_xy()
        if coords is None:
            self.status_var.set("Anchor cancelled.")
            return
        nx, ny = coords
        self.anchors.append({"mosaic": uv, "stage": stage, "nanosims": (float(nx), float(ny))})
        self._recompute_nanosims()
        self._refresh_trees()
        self.canvas.redraw()

    def _ask_nanosims_xy(self) -> Optional[Tuple[float, float]]:
        win = tk.Toplevel(self)
        win.title("NanoSIMS Anchor")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)

        x_var = tk.StringVar()
        y_var = tk.StringVar()
        result: Dict[str, Tuple[float, float]] = {}

        ttk.Label(win, text="NanoSIMS X").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Entry(win, textvariable=x_var, width=18).grid(row=0, column=1, padx=10, pady=(10, 4))
        ttk.Label(win, text="NanoSIMS Y").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=y_var, width=18).grid(row=1, column=1, padx=10, pady=4)

        buttons = ttk.Frame(win)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=10, pady=(8, 10))

        def ok() -> None:
            try:
                result["xy"] = (float(x_var.get()), float(y_var.get()))
            except Exception:
                messagebox.showwarning("NanoSIMS Anchor", "Enter numeric X and Y values.", parent=win)
                return
            win.destroy()

        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(buttons, text="OK", command=ok).pack(side="right", padx=(0, 8))
        win.bind("<Return>", lambda _e: ok())
        win.bind("<Escape>", lambda _e: win.destroy())
        win.after(50, lambda: win.focus_force())
        self.wait_window(win)
        return result.get("xy")

    def on_delete_anchor(self) -> None:
        sel = self.anchor_tree.selection()
        if not sel:
            return
        idx = int(sel[0].replace("A", "")) - 1
        if 0 <= idx < len(self.anchors):
            del self.anchors[idx]
        self._recompute_nanosims()
        self._refresh_trees()
        self.canvas.redraw()

    def on_clear_anchors(self) -> None:
        self.anchors.clear()
        self.stage_to_nanosims_A = None
        for pos in self.positions.values():
            pos["nano"] = None
        self._refresh_trees()
        self.canvas.redraw()
        self.status_var.set("NanoSIMS anchors cleared.")

    def _recompute_nanosims(self) -> None:
        if len(self.anchors) < 3:
            self.stage_to_nanosims_A = None
            for pos in self.positions.values():
                pos["nano"] = None
            self.status_var.set(f"{len(self.anchors)} anchor(s). Add at least 3 to update NanoSIMS coordinates.")
            return
        src = np.array([a["stage"] for a in self.anchors], dtype=float)
        dst = np.array([a["nanosims"] for a in self.anchors], dtype=float)
        try:
            self.stage_to_nanosims_A = affine_from_points(src, dst)
        except Exception as exc:
            self.stage_to_nanosims_A = None
            messagebox.showerror("Mapping", f"Could not compute NanoSIMS transform:\n{exc}")
            return
        for pos in self.positions.values():
            stage = pos.get("stage")
            pos["nano"] = apply_affine(self.stage_to_nanosims_A, stage) if stage is not None else None
        self.status_var.set(f"NanoSIMS coordinates updated from {len(self.anchors)} anchors.")

    def _refresh_trees(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for pid in sorted(self.positions):
            pos = self.positions[pid]
            stage = pos.get("stage")
            nano = pos.get("nano")
            self.tree.insert(
                "",
                "end",
                iid=pid,
                values=(
                    pid,
                    fmt_m(stage[0] if stage else None),
                    fmt_m(stage[1] if stage else None),
                    fmt_m(pos.get("z")),
                    fmt_ns(nano[0] if nano else None),
                    fmt_ns(nano[1] if nano else None),
                ),
            )

        for item in self.anchor_tree.get_children():
            self.anchor_tree.delete(item)
        for i, anchor in enumerate(self.anchors, start=1):
            stage = anchor["stage"]
            nano = anchor["nanosims"]
            self.anchor_tree.insert(
                "",
                "end",
                iid=f"A{i}",
                values=(f"A{i}", fmt_m(stage[0]), fmt_m(stage[1]), fmt_ns(nano[0]), fmt_ns(nano[1])),
            )

    def on_save_json(self) -> None:
        if not self.data:
            messagebox.showinfo("Save", "Load a mapping JSON first.")
            return
        data = copy.deepcopy(self.data)
        data["nanosims_conversion"] = {
            "anchors": [
                {
                    "mosaic": [float(a["mosaic"][0]), float(a["mosaic"][1])],
                    "confocal_stage": [float(a["stage"][0]), float(a["stage"][1])],
                    "nanosims": [float(a["nanosims"][0]), float(a["nanosims"][1])],
                }
                for a in self.anchors
            ],
            "stage_to_nanosims_A": self.stage_to_nanosims_A.tolist() if self.stage_to_nanosims_A is not None else None,
        }
        entries = ((data.get("positions") or {}).get("entries") or {})
        for pid, pos in self.positions.items():
            if pid not in entries:
                continue
            nano = pos.get("nano")
            if nano is None:
                entries[pid].pop("nanosims_off", None)
            else:
                entries[pid]["nanosims_off"] = {"x": float(nano[0]), "y": float(nano[1])}

        default_name = "converted.json"
        if self.json_path is not None:
            default_name = f"{self.json_path.stem}_nanosims.json"
        initial_dir = r"F:\ImSpector_Test\ImSpector_Test" if os.path.isdir(r"F:\ImSpector_Test\ImSpector_Test") else os.getcwd()
        out = filedialog.asksaveasfilename(
            title="Save converted JSON",
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not out:
            return
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save", f"Could not save JSON:\n{exc}")
            return
        messagebox.showinfo("Save", f"Saved converted JSON:\n{out}")


if __name__ == "__main__":
    app = NanoSimsConverter()
    app.mainloop()
