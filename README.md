# Confocal to NanoSIMS Coordinate Converter

A Streamlit web app for converting confocal overview/sample coordinates into NanoSIMS coordinates.

The app loads a mapping JSON and associated image tiles, reconstructs an overview image, lets the user define matching NanoSIMS anchor points, computes an affine transformation, and exports an updated JSON file containing converted NanoSIMS positions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use these settings:

- Repository: your GitHub repository URL
- Branch: `main` or the branch shown on GitHub
- Main file path: `app.py`

## How to use

1. Upload the mapping JSON.
2. Upload the image tiles individually, or upload a ZIP containing the tiles.
3. Click the overview image to choose an anchor point.
4. Enter the matching NanoSIMS X/Y coordinates.
5. Add at least 3 anchors.
6. Download the converted JSON.
