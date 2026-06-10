# NanoSIMS Converter — Replit version

This project wraps the original `convert.py` Tkinter GUI so it can be opened from Replit's **Run** button.

## Files

- `convert.py` — original application code.
- `main.py` — small Replit entry point that launches the GUI.
- `requirements.txt` — Python packages: `numpy` and `Pillow`.
- `.replit` — tells Replit to run `python main.py`.
- `replit.nix` — adds system packages needed for Tkinter and image libraries.

## How to use in Replit

1. Create a new Replit project.
2. Choose Python or import/upload this folder as a ZIP.
3. Make sure hidden files are included, especially `.replit`.
4. Click **Run**.
5. Upload your mapping JSON and image tile files into the Replit project file tree.
6. In the GUI, click **Load JSON** and select the uploaded mapping JSON.
7. After adding at least 3 NanoSIMS anchors, click **Save Converted JSON**.

## Important note about paths

Your original script has a Windows starting folder:

```python
F:\ImSpector_Test\ImSpector_Test
```

On Replit, that folder will not exist. The script already falls back to the current Replit workspace, so upload your JSON and image files into the project folder or subfolders.

If the JSON contains image paths from your old Windows computer, the script tries to find tiles by filename in the JSON folder or parent folder. For best results, upload the JSON and all tile images together in the same folder.

## If Tkinter does not open

Replit's Tkinter support can change. If the GUI does not open or you see `_tkinter` / display errors, ask Replit Agent:

> Configure this project to run a Python Tkinter GUI using Nix and VNC. The run command should be `python main.py`.

A more reliable alternative is GitHub Codespaces with noVNC, or converting this Tkinter GUI to a web app using Streamlit/NiceGUI.
