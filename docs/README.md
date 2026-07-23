# DeLux Project Page

Static GitHub Pages site for **DeLux: Cross-Modal Local Artifact Restoration in Video Using Neuromorphic Data**.

## Preview locally

```bash
cd docs
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Deploy on GitHub Pages

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `main` branch and the `/docs` folder.
5. Save. The page will be available at:

```
https://tremirre.github.io/event-sun-effects-remover/
```

## Assets

- Images and diagrams were copied from `paper/` and `data/figs/`, then compressed for the web.
- Videos were re-encoded to H.264 720p for faster loading on GitHub Pages.

## Updating content

- Edit `index.html` for structure and text.
- Edit `assets/css/style.css` for styling.
- Edit `assets/js/main.js` for interactivity (sliders, navigation, citation copy).
- Replace files in `assets/images/` or `assets/videos/` to update visuals.
