# Overview to NanoSIMS — Streamlit edition

A browser-based conversion of the original Tkinter utility.

## Features

- Upload a mapping JSON or a ZIP containing the JSON and referenced image tiles
- Render the image mosaic with two configurable channels
- Adjust channel visibility, intensity range and LUT
- Add corresponding NanoSIMS anchors by clicking the overview
- Fit a 2D affine transform from three or more anchors
- Preview converted sample coordinates
- Download the updated mapping JSON

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. In Streamlit Community Cloud, create an app from the repository.
4. Set the main file to `app.py`.

## Input format

A plain JSON upload loads coordinates, but remote browser apps cannot access local Windows paths stored in the JSON.

For image rendering, create a ZIP containing:

```text
project.zip
├── mapping.json
├── tile_001.tif
├── tile_002.tif
└── ...
```

The tile filenames must match the filenames referenced by the JSON. Folder paths are ignored and matching is done by filename.
