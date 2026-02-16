# Research: 2D image → 3D model/mesh (OBJ/STL/GLTF/PLY) usable in SketchUp

Date: 2026-02-15 (IST)

## Executive takeaways
- **Best open-weights quality (late-2025/2026) for single-image → textured mesh** is typically:
  - **TRELLIS.2 (Microsoft, ~4B)**: best open weights + **PBR** materials; but needs **Linux + NVIDIA ≥24GB VRAM**.
    Repo: https://github.com/microsoft/TRELLIS.2
  - **SAM 3D Objects (Meta)**: robust on real photos with clutter/occlusion, but requires **object masks** and **Linux + NVIDIA ≥32GB VRAM**.
    Repo: https://github.com/facebookresearch/sam-3d-objects
  - **Hunyuan3D 2.1 (Tencent)** / **Step1X-3D (StepFun)**: strong texture/PBR-ish pipelines but **very heavy VRAM (~27–29GB)**.
    Repos: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 • https://github.com/stepfun-ai/Step1X-3D
- **Best lightweight open-source baseline**: **TripoSR** (fast, simple; README suggests ~6GB VRAM on GPU).
  Repo: https://github.com/VAST-AI-Research/TripoSR
- **On Apple Silicon Mac (M3 Pro, low unified memory)**:
  - Most serious open-source pipelines assume **CUDA/NVIDIA** and remain impractical locally.
  - **Exception**: **TripoSR can run locally in CPU mode** and export **OBJ/GLB** (I tested it; see TripoSR section). Texture baking on macOS is currently painful.
  - For reliable **textured** meshes, use **hosted APIs** (Tripo3D, Meshy, Stability SF3D, Hyper3D Rodin, fal.ai Trellis) or a **cloud GPU**.
  - I did **not** find credible, maintained **CoreML/MLX ports** of these image→3D pipelines as of Feb 2026.

---

## SketchUp-friendly pipeline (practical)
1. Generate a mesh in **GLB/GLTF** or **OBJ**.
   - Prefer **GLB** when you have textures/materials (self-contained).
2. Open in **Blender** → cleanup:
   - Fix scale/orientation
   - Reduce polycount (SketchUp can choke on huge triangle counts)
   - Fix UVs/materials if needed
3. Export for SketchUp:
   - **DAE (Collada)** is often the most reliable import route
   - **OBJ** also works (keep `.mtl` + textures folder)
   - **STL** is geometry-only (no textures)

Notes:
- Many single-image pipelines output **vertex colors** rather than UV textures.
- PBR GLB materials (metal/rough) may not map 1:1 into SketchUp; exporting via Blender to DAE/OBJ is usually more predictable.

---

## Apple Silicon note: CoreML / MLX (Feb 2026)
- I did not find maintained **CoreML** or **MLX** ports for **TripoSR / SF3D / TRELLIS / Hunyuan3D / Step1X-3D**.
- Practical blockers are usually: model size, multi-stage pipelines, and custom ops (plus many repos being CUDA-first).
- The realistic Mac options today are:
  - **Run a lightweight model on CPU/MPS in PyTorch** (TripoSR is workable on CPU; MPS would need patching)
  - **Use hosted APIs** and treat the Mac as a post-processing + SketchUp workstation

---

## 1) `facebookresearch/sam-3d-objects` — what it does & fit
Repo: https://github.com/facebookresearch/sam-3d-objects

### What it is
- Reconstructs **pose + shape + texture + layout** for **masked objects** from a **single RGB image**.
- Designed for real-world photos (occlusions/clutter), supports multiple objects via multiple masks.
- Can export a **Gaussian Splat / point-based** representation and mesh-like outputs depending on pipeline.

### When it’s a good fit
- Your inputs are messy (phone photos, background clutter).
- You can provide a clean **mask** (SAM / manual mask / Photoshop).

### Practical constraints
- Heavy GPU requirements; commonly documented as needing **Linux + high VRAM NVIDIA**.
- Not a great target for Apple Silicon local runs today.

---

## 2) `microsoft/TRELLIS.2` — high-end open weights
Repo: https://github.com/microsoft/TRELLIS.2

### Why it’s interesting
- One of the strongest open-weight stacks for **textured assets with PBR-ish materials**.
- Better topology/material consistency than many “fast” models.

### Practical constraints
- Generally expects **Linux + NVIDIA** with significant VRAM.
- Not practical to install/test on a low-memory Mac without cloud GPU.

---

## 3) Hunyuan3D 2.1 / Step1X-3D — strong but heavy
- Hunyuan3D 2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Step1X-3D: https://github.com/stepfun-ai/Step1X-3D

### Notes
- Strong quality, often better textures than lightweight baselines.
- **VRAM hungry** (commonly reported high-20GB class).
- Best used via cloud GPU or a workstation.

---

## 4) TripoSR (VAST / Stability) — fastest open-weight baseline
Repo: https://github.com/VAST-AI-Research/TripoSR

### What it is
- Single-image → 3D mesh quickly.
- Can export **OBJ** and **GLB**.
- Optional texture baking path uses `xatlas` + OpenGL (`moderngl`).

### Why it’s useful
- It’s the easiest open-weights entry point (simple repo, fast inference).
- Even without perfect textures, it’s a good baseline for shape.

---

## ✅ TripoSR local install + CPU test on Mac (Apple Silicon, M3 Pro)
This is a practical, *tested* path for running TripoSR locally without CUDA.

### Repo used
- Clone: `/Users/oz/.openclaw/workspace/tmp_triposr`
- Requirements (upstream):
  - `trimesh==4.0.5` (too old for NumPy 2.x GLB export; see below)
  - `xatlas==0.0.9` (fails to build on macOS arm64 in my tests)
  - `moderngl==5.10.0` (used for texture baking)
  - `rembg` (background removal; needs onnxruntime)

### What worked
- ✅ **CPU inference works**.
- ✅ Exports:
  - **OBJ** export works (vertex-colored)
  - **GLB** export works after upgrading `trimesh` (NumPy 2.x compatibility)
- ✅ Background removal works after installing `onnxruntime` (rembg downloads `u2net.onnx` ~176MB on first run).

### Main macOS problems encountered (and fixes)
1) **`xatlas==0.0.9` build fails** on macOS arm64
- Failure was in CMake/pybind policy during wheel build.
- Practical workaround: **don’t use `--bake-texture`**, and avoid importing `xatlas` at runtime.

2) `rembg` failed with missing `onnxruntime`
- Fix: `pip install onnxruntime`

3) **GLB export failure due to NumPy 2.x**
- Upstream requirements pin `trimesh==4.0.5`, which is too old.
- Fix: `pip install -U trimesh`

4) `run.py` bug: when `--no-remove-bg`, output dir wasn’t created
- Symptoms: export fails because `output/<i>/` doesn’t exist.
- Fix: create dirs before export (I patched local clone).

### Local patch applied (in clone only)
I patched `tmp_triposr/run.py` to:
- **Lazy-import** `xatlas` / `moderngl` only when `--bake-texture` is actually requested.
- Ensure per-image output directories exist even when background removal is skipped (incl. when using a custom `--output-dir`).

### Minimal working setup (CPU)
```bash
cd /Users/oz/.openclaw/workspace/tmp_triposr
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel

# PyTorch CPU (Apple Silicon)
pip install torch torchvision torchaudio

# Core deps (avoid xatlas/moderngl if you won’t bake textures)
pip install omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 \
  rembg huggingface-hub imageio[ffmpeg] gradio

# Required by rembg
pip install onnxruntime

# Fix GLB export on NumPy 2.x
pip install -U trimesh

# torchmcubes from git (as in requirements.txt)
pip install git+https://github.com/tatsy/torchmcubes.git
```

### Run (CPU)
```bash
cd /Users/oz/.openclaw/workspace/tmp_triposr
source .venv/bin/activate

# simplest (positional image arg)
python run.py examples/chair.png --device cpu --mc-resolution 128

# faster iteration (skip background removal; lower marching-cubes resolution)
python run.py examples/chair.png \
  --device cpu \
  --mc-resolution 64 \
  --no-remove-bg \
  --model-save-format glb \
  --output-dir output_bench
```

Outputs go under the output directory (default `output/`) as `.../<index>/mesh.obj` or `mesh.glb`.

### Quick CPU benchmark (Mac M3 Pro, 1 image)
Measured locally on CPU with:
- `--mc-resolution 64`
- `--no-remove-bg`

Observed timings:
- Model init: ~6.0s
- Model inference: ~5.3s
- Mesh extraction: ~0.16s
- Export GLB: ~0.00–0.01s
- End-to-end wall time: ~15s
- Peak RAM (RSS): ~3.5GB

(Expect higher times for `--mc-resolution 128` and for background removal.)

### What didn’t work
- `--bake-texture` on macOS (blocked by `xatlas` build issues).
- MPS acceleration is not wired by default in upstream device logic (CUDA-first assumptions). It’s likely patchable, but wasn’t required to validate “does it run at all.”

---

## 4a) `threestudio` framework (incl. Zero-1-to-3 support)
Repo: https://github.com/threestudio-project/threestudio#zero-1-to-3-

### What it is
- A **research framework** for generating 3D content from:
  - **text prompts** (DreamFusion/Magic3D-style “score distillation” training)
  - **single images** (incl. **Zero-1-to-3 / Stable Zero123**-style guidance)
- This is **not** a single forward-pass “image→mesh in seconds” model.
  - Typical workflow is **optimization / training per object** (e.g., ~10k steps per run per their quickstart).

### Quality (practical)
- Can produce impressive results (especially with high-end guidance models), but:
  - Quality is sensitive to **prompts / configs / number of steps**.
  - Often needs **post-cleanup** (topology, UVs, texture seams) before SketchUp.

### Speed
- Expect **minutes → hours per asset**, depending on:
  - step count, resolution, chosen guidance model, and GPU.

### Hardware requirements
- threestudio’s installation docs explicitly assume **Linux + NVIDIA CUDA**:
  - README + docs say it’s tested on Ubuntu; requires **NVIDIA GPU** and **CUDA**.
  - Minimum: **≥6GB VRAM**; higher-quality configs can need **>20GB VRAM**.
  - Installation notes mention CUDA extensions (e.g., `tiny-cuda-nn`) and CUDA rasterizer options.

### Output formats (relevant for SketchUp)
- Export supports **OBJ**:
  - **OBJ+MTL** (textured) or
  - **OBJ with vertex colors**
  (per `Export Meshes` section in threestudio README)

### Mac M3 Pro feasibility
- **Not feasible locally** on an M3 Pro (no CUDA; many dependencies are CUDA-first).
- If you want to use threestudio anyway:
  - run it on a **cloud NVIDIA GPU** (or Colab) and export **OBJ** for Blender/SketchUp.

---

## 4b) `cvlab-columbia/zero123` (Zero-1-to-3) — novel views → (optional) 3D
Repo: https://github.com/cvlab-columbia/zero123

### What it is
- Primarily a **novel view synthesis** model: given 1 image, generate plausible views from new camera angles.
- It becomes “image→3D” only when paired with a **separate reconstruction stage** (NeRF/voxels/SJC/SDS).

### Quality (practical)
- Strong at generating multi-view-consistent-ish views for simple objects.
- Still prone to:
  - hallucinated unseen surfaces
  - texture inconsistency
  - geometry ambiguity (it’s not directly predicting a mesh)

### Speed
- Novel view generation is diffusion-based (not instant).
- 3D reconstruction path is **iterative optimization** (e.g., they show ~10k steps for SJC-based reconstruction).

### Hardware requirements (from their README)
- They note their Gradio demo uses **~22GB VRAM** (i.e., RTX 3090/4090 class).
- They also say they tested installation on **Ubuntu 20.04 + NVIDIA Ampere**.

### Output formats
- Base model output: **images** (novel views).
- For 3D, their repo points to additional pipelines and mentions exporting a mesh via marching cubes in their 3D reconstruction code.
  - So practically you can end up with a **mesh** (then export to OBJ/PLY depending on the reconstruction pipeline you use).

### Mac M3 Pro feasibility
- **Not feasible locally** in a practical sense:
  - The repo assumes CUDA/Linux for the demo-scale workloads.
  - CPU/MPS would be extremely slow and/or blocked by CUDA-specific deps.
- Recommended Mac workflow:
  - Use the **Hugging Face Space** for quick view-generation tests: https://huggingface.co/spaces/cvlab/zero123-live
  - Or use a cloud GPU and treat your Mac as the Blender/SketchUp post-processing machine.

---

## 5) Hosted APIs / SaaS that do this well (paid is OK)

### Tripo3D API (Tripo)
- API docs + OpenAPI schema: https://platform.tripo3d.ai/docs/schema
  - **Base URL (per schema)**: `https://api.tripo3d.ai/v2/openapi`
  - Key endpoints (per schema):
    - `POST /upload` (upload an image)
    - `POST /task` (create task; includes `image_to_model`)
    - `GET /task/{task_id}` (poll status + get outputs)
    - `GET /user/balance`
- Pricing/free tier (Tripo Studio): https://www.tripo3d.ai/pricing
  - **Basic = $0/month**
  - **300 credits/month**, **1 concurrent task**, limited downloads/storage, 1-day history
- Notes:
  - I did **not** run live API calls (needs an API key + login), but the OpenAPI schema is complete enough to integrate.

### Stability AI — “Stable Fast 3D (SF3D)” API
- API reference (REST v2beta): https://platform.stability.ai/docs/api-reference
- Pricing: https://platform.stability.ai/pricing
  - Pricing is in credits; **1 credit = $0.01**.
- Endpoint (from Stability API reference):
  - `POST https://api.stability.ai/v2beta/3d/stable-fast-3d`
  - Request body: `multipart/form-data`
    - required: `image`
    - optional: `texture_resolution` (512/1024/2048), `foreground_ratio`, `remesh` (none/triangle/quad), `vertex_count`
  - Response: **binary GLB** (glTF)
  - **Cost**: **10 credits** per successful generation (≈ **$0.10**)
- Note: Stability’s public pricing page doesn’t mention a “free tier” (assume you’ll need to buy credits).

### Meshy (API)
- Docs: https://docs.meshy.ai/en
  - **Base URL**: `https://api.meshy.ai`
- Image → 3D endpoints (docs): https://docs.meshy.ai/en/api/image-to-3d
  - `POST /openapi/v1/image-to-3d` (create task)
  - `GET /openapi/v1/image-to-3d/:id` (poll)
  - `GET /openapi/v1/image-to-3d/:id/stream` (SSE)
- Pricing per call (API docs): https://docs.meshy.ai/en/api/pricing
  - Image→3D:
    - **Meshy-6 / latest / lowpoly**: 20 credits (no texture) or 30 credits (with texture)
    - **Other models**: 5 credits (no texture) or 15 credits (with texture)
- Subscription/free tier notes (Meshy web pricing): https://www.meshy.ai/pricing
  - **Free plan** shows **100 monthly credits**, but **API & plugins access** is listed under **Pro**.
- ✅ Test mode (no signup needed)
  - Quickstart: https://docs.meshy.ai/en/api/quick-start
  - Test key shown in docs: `msy_dummy_api_key_for_test_mode_12345678`
  - Behavior: no credits consumed; returns deterministic sample results.
  - Verified working locally:
    - `POST https://api.meshy.ai/openapi/v1/image-to-3d` returned a task id
    - `GET https://api.meshy.ai/openapi/v1/image-to-3d/{id}` returned `SUCCEEDED` + downloadable `model_urls` (GLB/FBX/USDZ/etc)

### Hyper3D Rodin API
- API overview/spec: https://developer.hyper3d.ai/api-specification/overview
- Rodin generation endpoint: https://developer.hyper3d.ai/api-specification/rodin-generation
- Supports **1–5 images**; exports include **GLB/OBJ/FBX** etc; quality presets.

### fal.ai hosted TRELLIS
- Some workflows use fal.ai to call Trellis/Trellis.2-like backends without local GPU.
- Verify exact model/version in fal’s model list before committing.

### Kaedim (commercial)
- Strong for “production-ish” meshes (often with human-in-the-loop options), popular in game asset pipelines.
- API availability via enterprise/dev programs; check vendor docs/quoting.

### Luma AI
- Great for photogrammetry-like capture → 3D from phone video.
- Public “Genie” programmatic API is not clearly documented as of early 2026; treat as primarily app/workflow-driven.

### (Bonus) Free/cheap hosted inference: Hugging Face Spaces + Replicate
These are useful when you want to test quality quickly without wiring up a paid API.

**Hugging Face Spaces (often free, but can be rate-limited / sleeping)**
- Stability’s Spaces exist for:
  - Stable Fast 3D: https://huggingface.co/spaces/stabilityai/stable-fast-3d
  - TripoSR: https://huggingface.co/spaces/stabilityai/TripoSR
- Caveat: Spaces can be broken/paused depending on dependencies and queue load; treat as “best effort.”

**Replicate (paid, but easy + transparent pricing; good for heavy models)**
- Example confirmed model:
  - Hunyuan 3D 3.1: https://replicate.com/tencent/hunyuan-3d-3.1
    - Inputs include a prompt + optional **image**; can enable **PBR**.
    - Pricing shown on the model page: **$0.16 per unit** (see its Pricing section).
- Replicate platform pricing reference: https://replicate.com/pricing

---

## 6) Practical recommendation (what to try first)

### If you want the fastest path on your current machine (Mac M3 Pro)
1) Use a hosted API that returns GLB directly:
   - **Stability SF3D** (fast baseline) or **Tripo3D API** (flexible exports)
   - Meshy is also good if you already have a Pro plan / credits
2) Open resulting GLB in **Blender** → reduce polys → export DAE/OBJ for SketchUp.

### If you have access to an NVIDIA GPU box (or can rent one)
- Start with **TripoSR**.
- If you want better textures/PBR and have **24GB+ VRAM**:
  - **TRELLIS.2**
  - **Hunyuan3D 2.1 / Step1X-3D**
- For messy photos and you can provide masks:
  - **SAM 3D Objects** (plan for 32GB VRAM class GPU)

### When to abandon “single-image generative” and use classic photogrammetry
If accuracy matters (dimensions, exact geometry), shoot **30–200 photos** and run **COLMAP/Meshroom**. Single-image methods are “plausible 3D,” not guaranteed true shape.

---

## 7) Getting started (practical): image → 3D model → SketchUp (Mac workflow)

### Step 0 — Pick/prepare the input image (biggest quality lever)
- Use a **single, centered object** with a clear silhouette.
- Prefer **plain background**.
- Avoid heavy occlusion, motion blur, and extreme wide-angle distortion.

If you can, do one of these first:
- Remove background (remove.bg / PhotoRoom / Photoshop / `rembg`)
- Or crop tightly around the object

---

### Option A — “Try it now” (no signup): Meshy API test mode
Meshy’s test mode key returns sample GLB/FBX/USDZ without consuming credits.

```bash
curl -X POST 'https://api.meshy.ai/openapi/v1/image-to-3d' \
  -H 'Authorization: Bearer msy_dummy_api_key_for_test_mode_12345678' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_url":"https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png",
    "ai_model":"latest",
    "model_type":"standard",
    "should_texture":false
  }'
# → returns an id like: 019c....

curl -X GET 'https://api.meshy.ai/openapi/v1/image-to-3d/<ID>' \
  -H 'Authorization: Bearer msy_dummy_api_key_for_test_mode_12345678'
# → returns status + model_urls.glb/fbx/usdz
```

Use this to validate your downstream pipeline (Blender → SketchUp import) before paying for anything.

---

### Option B — Best “paid but simple” baseline: Stability Stable Fast 3D (SF3D)
1) Create a Stability API key
2) Run:

```bash
export STABILITY_API_KEY='...'

curl -X POST 'https://api.stability.ai/v2beta/3d/stable-fast-3d' \
  -H "authorization: Bearer $STABILITY_API_KEY" \
  -H 'accept: model/gltf-binary' \
  -F 'image=@input.png' \
  -F 'texture_resolution=1024' \
  --output out.glb
```

- Expect **~$0.10** per successful generation.
- Output is a **GLB** you can immediately open in Blender.

---

### Option C — Local (free) on Apple Silicon: TripoSR CPU mode
Great for quick experimentation when you don’t want to pay per call.

```bash
cd /Users/oz/.openclaw/workspace/tmp_triposr
source .venv/bin/activate

python run.py examples/chair.png --device cpu --mc-resolution 128
```

Notes:
- Texture baking (`--bake-texture`) may fail on macOS due to `xatlas`.
- If you need clean UV textures for SketchUp, a hosted API is usually faster than fighting macOS build deps.

---

### Step 3 — Clean up / reduce polys in Blender (recommended)
1) Import `out.glb` in Blender.
2) Apply transforms:
   - Select mesh → `Ctrl+A` → Apply Rotation & Scale
3) Reduce complexity:
   - Add **Decimate** modifier (start ratio 0.2–0.5)
4) Fix normals/shading:
   - Mesh → Normals → Recalculate Outside
   - Enable Auto Smooth if needed

Target guidance for SketchUp stability:
- Try to stay under **50k–200k triangles** per asset (depends on your overall scene).

---

### Step 4 — Export a SketchUp-friendly format
Import support varies by SketchUp version/plugins, but these routes are usually reliable:

**Route 1: Collada (.dae)**
- Blender: File → Export → Collada (.dae)
- SketchUp: File → Import → Collada

**Route 2: OBJ (+ MTL + textures folder)**
- Blender: File → Export → Wavefront (.obj)
- Ensure materials are written and textures are copied.

Tips:
- GLB PBR materials don’t always map perfectly into SketchUp.
- If textures look wrong, re-check UVs in Blender and re-export.

---

### Step 5 — If you need accuracy (not just plausibility)
If this is for architecture/interiors and you need real dimensions:
- Shoot 30–200 images/video around the object
- Reconstruct with COLMAP/Meshroom/RealityCapture
- Then decimate + export to SketchUp
