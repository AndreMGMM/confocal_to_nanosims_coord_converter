# Confocal to NanoSIMS Coordinate Converter

A Streamlit web app for converting confocal overview/sample coordinates into NanoSIMS coordinate space.

## Features

- Upload a mapping JSON file.
- Upload image tiles individually or as a ZIP archive.
- Reconstruct the overview mosaic image.
- Adjust two-channel color display.
- Zoom the overview in/out.
- Click overview features to place anchor points.
- Display anchors directly on the overview image.
- Compute an affine transform from at least three NanoSIMS anchors.
- Export an updated JSON file with `nanosims_off` coordinates.

## Deploy on Streamlit Community Cloud

Use these settings:

```text
Repository: your GitHub repository URL
Branch: main or the branch shown on GitHub
Main file path: app.py
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
