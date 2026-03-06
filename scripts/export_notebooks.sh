#!/usr/bin/env bash
set -e

THEME_CSS="notebooks/blog-theme.css"

for nb in notebooks/*.py; do
  name=$(basename "$nb" .py)
  marimo export html-wasm "$nb" -o "static/notebooks/$name" --mode run

  # Inject the blog theme CSS into the exported HTML so the notebook
  # shares the same typography as the PaperMod site.
  if [ -f "$THEME_CSS" ]; then
    python3 - "$name" <<'EOF'
import sys
name = sys.argv[1]
html_path = f"static/notebooks/{name}/index.html"
css_path  = "notebooks/blog-theme.css"

with open(html_path, encoding="utf-8") as f:
    html = f.read()
with open(css_path, encoding="utf-8") as f:
    css = f.read()

tag = f"<style id=\"blog-theme\">\n{css}\n</style>"
html = html.replace("</head>", f"{tag}\n</head>", 1)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Blog theme CSS injected into static/notebooks/{name}/index.html")
EOF
  fi
done
