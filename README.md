# Image → 3D Model Pipeline

Research and tools for converting 2D images into 3D meshes (OBJ/GLB) usable in SketchUp and Blender.

## Quick Start: TripoSR on Mac (CPU)

```bash
./setup-triposr.sh          # uses uv for fast dependency installation
source triposr/.venv/bin/activate
cd triposr
python run.py examples/chair.png --device cpu --mc-resolution 128
```

> Requires [uv](https://github.com/astral-sh/uv). The script will install it automatically if missing.

Output: `triposr/output/0/mesh.obj` (or `.glb` with `--model-save-format glb`)

### Options

| Flag | Description | Default |
|---|---|---|
| `--device cpu` | Force CPU (auto-detected on Mac) | `cuda:0` |
| `--mc-resolution N` | Marching cubes resolution (64=fast, 256=detailed) | `256` |
| `--no-remove-bg` | Skip background removal (faster) | off |
| `--model-save-format glb` | Export GLB instead of OBJ | `obj` |
| `--output-dir DIR` | Custom output directory | `output/` |

### Performance (Mac M3 Pro, CPU, mc-resolution 64)

- ~15s end-to-end, ~3.5GB peak RAM

## Research

See [RESEARCH.md](RESEARCH.md) for a comprehensive evaluation of open-source models, hosted APIs, and the full Mac → Blender → SketchUp pipeline.

## What's Next

- [ ] Test hosted APIs (Stability SF3D, Tripo3D) for textured meshes
- [ ] Try MPS acceleration on Apple Silicon
- [ ] Blender batch cleanup script
