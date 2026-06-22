# Overview to NanoSIMS — Streamlit edition v3

## Improvements in v3

- Header spacing prevents the Streamlit toolbar from covering the title.
- TIFF tiles are decoded once when the project loads.
- The expensive mosaic is rendered only when **Apply display settings** is pressed.
- Explicit **Add anchor** mode in the Anchors tab.
- NanoSIMS X and Y are entered in a one-row editor and saved with a dedicated button.
- Large yellow crosshair anchors labelled A1, A2, etc.
- Green converted NanoSIMS coordinate columns.
- Plotly viewer supports mouse-wheel zoom, click-and-drag panning, and double-click reset.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
