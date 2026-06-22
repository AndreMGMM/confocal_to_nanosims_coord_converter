# Overview to NanoSIMS — Streamlit edition v6.1

A browser-based tool for converting confocal overview coordinates to NanoSIMS coordinates using user-defined affine anchors.

## Main features

- Load a project as one ZIP file, or select the mapping JSON and all referenced image tiles together.
- Render two image channels with configurable visibility, intensity limits and colour LUTs.
- Cache decoded TIFF tiles and the rendered mosaic to reduce unnecessary image processing.
- Interactive HTML canvas viewer:
  - mouse-wheel zoom centred on the cursor;
  - click-and-drag panning;
  - double-click to reset the view;
  - responsive width that follows the browser window;
  - adjustable canvas height in the **Viewer** sidebar block.
- Optional ROI and anchor overlays.
- Large yellow anchors labelled `A1`, `A2`, and so on.
- Add anchors by clicking the overview and entering the corresponding NanoSIMS X and Y values.
- Fit a two-dimensional affine transformation from three or more anchors.
- Preview converted coordinates and export an updated JSON file.

## Changes in v6.1

- Fixed the Add Anchor workflow so one press activates anchor-placement mode.
- Increased the default canvas height from 720 to 820 pixels.
- Added a **Canvas height** slider ranging from 600 to 1200 pixels.
- Increased the width allocated to the overview panel.
- Documented the custom interactive canvas and project input format.

## Interaction controls

| Action | Control |
|---|---|
| Zoom | Mouse wheel over the overview |
| Pan | Click and drag the overview |
| Reset view | Double-click the overview |
| Place an anchor | Press **Add anchor**, then click the desired feature |
| Resize vertically | Use **Viewer → Canvas height** in the sidebar |
| Resize horizontally | Widen the browser window or collapse the sidebar |

## Project input

### Recommended: ZIP upload

```text
project.zip
├── mapping.json
├── tile_001.tif
├── tile_002.tif
└── ...
```

The image filenames must correspond to the paths referenced in the JSON. Folder differences and Windows path separators are handled by matching filenames and filename stems.

### Multiple-file upload

You can instead select the JSON and all image tiles together in the file uploader.

A JSON file uploaded without its image tiles can still provide coordinate information, but the overview image cannot be rendered because a web application cannot access local paths such as `C:\...` or `F:\...` stored in the JSON.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the complete project, including:
   - `app.py`
   - `requirements.txt`
   - the entire `interactive_canvas` directory
3. Create a new app in Streamlit Community Cloud.
4. Select `app.py` as the main file.

The `interactive_canvas/index.html` file is required. Do not omit the directory when copying or deploying the project.

## Display performance

Changing channel visibility, LUT or intensity settings does not immediately rebuild the mosaic. Press **Apply display settings** to render the updated image. This deliberate step avoids repeatedly decoding and compositing image data during normal interaction.

Zooming, panning and drawing markers occur inside the browser canvas and do not rebuild the TIFF mosaic.

## Coordinate conversion

At least three anchors are required to calculate the affine transformation. More than three anchors are supported and are fitted using least squares.

The downloaded JSON contains:

- the selected anchor points;
- the fitted stage-to-NanoSIMS affine matrix;
- converted `nanosims_off` coordinates for the available positions.
