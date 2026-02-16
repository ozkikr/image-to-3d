#!/usr/bin/env bash
# Setup script for TripoSR on macOS (Apple Silicon / CPU)
# Clones the repo, patches for macOS compatibility, and installs deps.
set -euo pipefail

REPO_DIR="triposr"

echo "==> Cloning TripoSR..."
if [ -d "$REPO_DIR" ]; then
  echo "    $REPO_DIR already exists, skipping clone."
else
  git clone https://github.com/VAST-AI-Research/TripoSR.git "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install -U pip setuptools wheel -q

# PyTorch (CPU / Apple Silicon MPS)
pip install torch torchvision torchaudio -q

# Core deps (skip xatlas — fails to build on macOS arm64)
pip install \
  omegaconf==2.3.0 \
  Pillow==10.1.0 \
  einops==0.7.0 \
  "transformers>=4.35.0" \
  rembg \
  huggingface-hub \
  "imageio[ffmpeg]" \
  gradio \
  -q

# rembg needs onnxruntime
pip install onnxruntime -q

# Fix GLB export on NumPy 2.x (upstream pins trimesh too old)
pip install -U trimesh -q

# torchmcubes (marching cubes on CPU/CUDA)
pip install git+https://github.com/tatsy/torchmcubes.git -q

echo ""
echo "==> Patching run.py for macOS compatibility..."

# Patch 1: Lazy-import xatlas/moderngl (only needed for --bake-texture)
# Patch 2: Ensure output dirs exist even with --no-remove-bg
# We apply patches only if they haven't been applied yet.

if ! grep -q "Lazy import so non-textured" run.py 2>/dev/null; then
  python3 - <<'PATCH_SCRIPT'
import re

with open("run.py", "r") as f:
    content = f.read()

# Add os.makedirs for --no-remove-bg case (before "Running model" log line)
# Look for the loop that processes each image
old_loop = '''    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")'''

new_loop = '''    logging.info(f"Running image {i + 1}/{len(images)} ...")

    # Ensure per-image output directory exists even when --no-remove-bg
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    timer.start("Running model")'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)

# Make xatlas/bake_texture import lazy (wrap in try/except inside --bake-texture block)
old_bake = '''    if args.bake_texture:
        import xatlas'''

new_bake = '''    if args.bake_texture:
        # Lazy import so non-textured (vertex color) export can run without xatlas/moderngl.
        try:
            import xatlas'''

if old_bake in content and "Lazy import" not in content:
    content = content.replace(old_bake, new_bake)
    # Add except block after the xatlas import line
    content = content.replace(
        "        from tsr.bake_texture import bake_texture\n",
        "            from tsr.bake_texture import bake_texture\n"
        "        except Exception as e:\n"
        '            raise RuntimeError(\n'
        '                "Texture baking requested, but xatlas/moderngl dependencies are not available. "\n'
        '                "Try pip install xatlas moderngl, or run without --bake-texture."\n'
        "            ) from e\n\n"
    )

with open("run.py", "w") as f:
    f.write(content)
PATCH_SCRIPT
  echo "    Patches applied."
else
  echo "    Already patched, skipping."
fi

echo ""
echo "==> Done! To use:"
echo "    source $REPO_DIR/.venv/bin/activate"
echo "    cd $REPO_DIR"
echo "    python run.py examples/chair.png --device cpu --mc-resolution 128"
echo ""
echo "    Output will be in output/0/mesh.obj"
