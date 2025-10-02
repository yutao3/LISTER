# LISTER

**LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction**

LISTER is a research-oriented open-source toolbox for reconstructing lunar surface topography from LROC NAC images. It takes either a photogrammetric DTM or the LOLA DTM as reference, and provides an automated and modular framework for generating high-resolution digital terrain models (DTMs) using integrated pre-trained Monocular Depth Estimation networks and an end-to-end reconstruction pipeline.

---

## Features

- End-to-end automated pipelines for generating DTMs and surface reconstructions  
- Support for both **high-resolution photogrammetric DTMs** and **LOLA DTMs** as reference  
- Integration of **dense U-Net**, **vision transformers** (NeWCRFs, DepthFormer) and **diffusion-based** monocular depth estimation models (Marigold E2E-FT, Stable Diffusion E2E-FT)  
- Scripts for **downloading**, **preprocessing**, and **converting** raw LROC NAC images  
- Tools for **brightness adjustment**, **quality filtering**, **georeferencing**, **mosaicking**, and **reprojection**  
- Modular design allowing standalone use of helper scripts and validation utilities  

---

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Models & Weights](#models--weights)  
4. [Data Preparation](#data-preparation)  
5. [Usage](#usage)  
   - [Automated Pipelines](#automated-pipelines)  
   - [Helper & Utility Scripts](#helper--utility-scripts)  
   - [Validation Scripts](#validation-scripts)  
6. [Outputs](#outputs)  
7. [Limitations & Notes](#limitations--notes)  
8. [Acknowledgements](#acknowledgements)  
9. [License](#license)  
10. [Citation](#citation)  
11. [Contact](#contact)

---

## Requirements

- **Python**: 3.8 or newer (3.10+ recommended)  
- **Core Python libraries**: `numpy`, `scipy`, `Pillow`, `opencv-python`, `shapely`, `pyproj`, `affine`, `pandas`, `tqdm`, `requests`, `pyyaml`  
- **Geospatial**:  
  - `gdal` (Python bindings) and GDAL command-line tools (`gdalwarp`, `gdal_translate`) — recommended  
  - or/and `rasterio` for reading/writing rasters where appropriate  
- **Deep learning**:  
  - `torch`, `torchvision` (GPU strongly recommended)  
  - `diffusers`, `transformers`, `huggingface-hub` (for diffusion models)  
  - `timm`, `einops`, `kornia` (for transformer/backbone utilities)  
- **External tools (recommended)**:  
  - **NASA Ames Stereo Pipeline (ASP)** for robust mosaicking and DEM operations: <https://github.com/NeoGeographyToolkit/StereoPipeline>  
- **Hardware**: CUDA-capable GPU with sufficient VRAM for diffusion-based inference and tiling. Ample disk space (tens of GB) for raw inputs, intermediates, and model checkpoints.

> **GDAL on Python** can be system-dependent. Confirm installation with:
> ```bash
> gdalinfo --version
> python -c "from osgeo import gdal; print(gdal.__version__)"
> ```

---

## Installation

Clone this repository:
```bash
git clone https://github.com/yutao3/LISTER.git
cd LISTER
```

Install Python dependencies (preferably inside a virtual environment):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install system packages as needed (e.g., GDAL, ASP). Ensure `gdalwarp`, `gdal_translate`, and (optionally) ASP tools like `dem_mosaic` are on your `PATH`.

---

## Models & Weights

Pre-trained weights and diffusion networks are **not** included due to size limits. Download them from Google Drive and place them under a local directory such as `weights/` (or any path you configure).

- **Google Drive (all weights & diffusion networks)**  
  <https://drive.google.com/drive/folders/1uQZtQKEiKxk3WJoZn2wMLVHRkegJQU6G?usp=drive_link>

Included/expected families of models:
- **Diffusion-based MDE** (based on VCI’s `diffusion-e2e-ft`): *Marigold E2E‑FT*, *Stable Diffusion E2E‑FT*  
- **Dense U-Net** (DenseDepth-derived)  
- **Vision Transformers**: *NeWCRFs*, *DepthFormer*

> Ensure your command-line options (below) point to the correct model folders and checkpoint filenames. Run `--help` on the scripts to see the exact argument names supported in this repository.

---

## Data Preparation

1. **Download LROC NAC** (if needed):  
   Use the provided utilities to fetch NAC EDRs by ID and to keep a current manifest.
   - `get_latest_lroc_nac_list_and_merge.py` — fetch latest NAC image list & merge metadata  
   - `get_lroc_nac.py` — download NAC EDRs by image ID(s)

2. **Convert EDR → GeoTIFF**:  
   - `prepare_geotiff_lroc_nac.py` — converts raw LROC NAC EDR to GeoTIFF suitable for downstream steps.  
     Example:
     ```bash
     python prepare_geotiff_lroc_nac.py \
       --input /path/to/EDR/ \
       --output /path/to/nac_geotiffs/
     ```

3. **Brightness & Quality (optional but recommended)**:  
   - `pre_adjust_brightness.py` — normalise exposure for more stable inference  
   - `check_low_quality_robust.py` — flag/remove problematic images before processing

4. **Reference DTM**: choose one of:  
   - **High-resolution photogrammetric DTM** (preferred for best local accuracy)  
   - **LOLA DTM** (for broad coverage; used by MSP pipelines)

> **Projection maps** are provided (`lroc_equator.map`, `lroc_northpole.map`, `lroc_southpole.map`) for appropriate map-projection handling near equator/poles.

---

## Usage

### Automated Pipelines

> The following examples illustrate the typical command patterns. For the **full and authoritative list of flags supported by your local scripts**, run `--help` on each script (e.g., `python LISTER_autoDTM.py --help`).

#### 1) Photogrammetric DTM as reference (recommended)
Main entry point: **`LISTER_autoDTM.py`**

```bash
python LISTER_autoDTM.py \
  --input-dir /path/to/nac_geotiffs \
  --dtm /path/to/highres_photogrammetric_dtm.tif \
  --weights-dir /path/to/weights \
  --model {denseunet|newcrfs|depthformer|marigold|sd_e2eft} \
  --tile-size 1024 --overlap 64 \
  --device cuda:0 \
  --out /path/to/output_dir
```
Typical options you may find useful:
- `--input-dir`: directory with prepared NAC GeoTIFFs (after EDR→GeoTIFF conversion)  
- `--dtm`: high-res photogrammetric DTM to register outputs against  
- `--weights-dir`: directory containing the downloaded model checkpoints  
- `--model`: which pre-trained model family to use for MDE/normal inference  
- `--tile-size`, `--overlap`: tiling parameters for large images  
- `--device`: CPU or CUDA device string  
- `--out`: output directory

#### 2) LOLA DTM as reference
Main entry points: **`LISTER_autoDTM_MSP.py`** and **`LISTER_autoDTM_MSP_light.py`**

```bash
python LISTER_autoDTM_MSP.py \
  --input-dir /path/to/nac_geotiffs \
  --lola-dtm /path/to/LOLA_global.tif \
  --weights-dir /path/to/weights \
  --model {denseunet|newcrfs|depthformer|marigold|sd_e2eft} \
  --tile-size 1024 --overlap 64 \
  --device cuda:0 \
  --out /path/to/output_dir
```

The *light* variant typically reduces compute/memory or applies simplified post-processing:
```bash
python LISTER_autoDTM_MSP_light.py \
  --input-dir /path/to/nac_geotiffs \
  --lola-dtm /path/to/LOLA_global.tif \
  --weights-dir /path/to/weights \
  --model sd_e2eft \
  --tile-size 768 --overlap 48 \
  --device cuda:0 \
  --out /path/to/output_dir
```

> Some pipelines optionally use **NASA Ames Stereo Pipeline** tools (e.g., `dem_mosaic`) to mosaic tiles robustly. Ensure ASP is installed and visible on your PATH if mosaicking via ASP is requested.

### Helper & Utility Scripts

The following scripts can be used independently or are invoked internally by the automated pipelines. Use `--help` on each to see all options supported by your local clone.

- **`get_latest_lroc_nac_list_and_merge.py`**  
  Fetch or update the NAC image list and merge metadata fields into a single reference file (CSV/JSON).  
  ```bash
  python get_latest_lroc_nac_list_and_merge.py \
    --out manifest.csv
  ```

- **`get_lroc_nac.py`**  
  Download LROC NAC images (EDR) by passing one or more image IDs or a list file.  
  ```bash
  python get_lroc_nac.py \
    --ids M1234567890RE M1234567891LE \
    --out /data/LROC_EDR/
  # or
  python get_lroc_nac.py --list ids.txt --out /data/LROC_EDR/
  ```

- **`prepare_geotiff_lroc_nac.py`**  
  Convert raw EDRs to processing-ready GeoTIFFs; handles map-projection and metadata.  
  ```bash
  python prepare_geotiff_lroc_nac.py \
    --input /data/LROC_EDR/ \
    --output /data/nac_geotiffs/ \
    --projection-map lroc_equator.map    # or lroc_northpole.map / lroc_southpole.map
  ```

- **`pre_adjust_brightness.py`**  
  Apply brightness normalisation to NAC images.  
  ```bash
  python pre_adjust_brightness.py \
    --input /data/nac_geotiffs/ \
    --output /data/nac_geotiffs_bright/
  ```

- **`check_low_quality_robust.py`**  
  Perform robust quality checks; optionally write a filtered list or flags.  
  ```bash
  python check_low_quality_robust.py \
    --input /data/nac_geotiffs/ \
    --out quality_report.csv \
    --min-contrast 0.05
  ```

- **`georeference.py`**, **`make_tiles.py`**, **`inference.py`**, **`postprocess*.py`**, **`full_chain.py`**  
  Lower-level building blocks used within the automated pipelines. For advanced use or custom experiments, inspect `--help` on each to run them step-by-step (tiling → inference → post-processing → georeference → mosaic).

### Validation Scripts

The `validation/` folder contains utilities to evaluate/compare generated DTMs against references and to compute statistics, profiles, or maps. Exact script names and options may vary; common patterns look like:

```bash
# Example: compute elevation error statistics vs. a reference DTM
python validation/validate_against_reference.py \
  --pred /path/to/LISTER_DTM.tif \
  --ref  /path/to/reference_DTM.tif \
  --mask /path/to/valid_mask.tif \
  --out  validation_report.json
```

```bash
# Example: generate hillshade/contours/quality overlays for quick inspection
python validation/visualise_products.py \
  --dtm /path/to/LISTER_DTM.tif \
  --image /path/to/NAC_image.tif \
  --out /path/to/figures/
```

> Please run `--help` on each script in `validation/` to see all supported arguments in your copy of the repository.

---

## Outputs

Typical outputs include:
- **High-resolution DTM** in GeoTIFF aligned to the chosen reference (photogrammetric or LOLA)  
- **Surface normal maps** or intermediate depth tiles (depending on the model used)  
- **Mosaics** of tiles (optionally via **NASA ASP** tools for robust blending)  
- **Quality maps**, logs, and optional diagnostic artefacts  
- **Intermediate files** (brightness-adjusted inputs, tiles, masks) which you may delete after validation to save space

Directory structure example:
```
outputs/
  dtm/
  normals/
  tiles/
  logs/
  qc/
```

---

## Limitations & Notes

- **GPU recommended**: Diffusion-based models are expensive to run on CPU.  
- **Registration assumptions**: The approach assumes reasonable alignment between NAC imagery and the chosen reference DTM; severe mis-registrations degrade results.  
- **Polar handling**: Use appropriate projection maps near the poles (`lroc_northpole.map`, `lroc_southpole.map`).  
- **Large data volumes**: Expect large intermediates; clean up when finished.  
- **Exact CLI flags**: Minor differences may exist between clones/branches. Use `--help` on each script to confirm arguments for your environment.

---

## Acknowledgements

This repository integrates or builds upon the following works and toolkits. Please consult their licenses before use:

- **DenseDepth (Dense U-Net)** — <https://github.com/ialhashim/DenseDepth>  
- **NeWCRFs (Vision Transformers for MDE)** — <https://github.com/aliyun/NeWCRFs>  
- **DepthFormer (Vision Transformers for MDE)** — <https://github.com/ashutosh1807/Depthformer>  
- **Diffusion E2E-FT (Marigold & Stable Diffusion E2E-FT)** — <https://github.com/VisualComputingInstitute/diffusion-e2e-ft>  
- **NASA Ames Stereo Pipeline (mosaicking/DEM tools)** — <https://github.com/NeoGeographyToolkit/StereoPipeline>  
- **GDAL, Rasterio, PyTorch, diffusers, transformers, timm, einops**, and the broader open-source geospatial & ML ecosystem.

> **Funding acknowledgement**: The work was carried out under a programme of and funded by the **European Space Agency (ESA)**.

---

## License

Unless otherwise noted, this repository is released under the **Apache-2.0 License** (see `LICENSE`).  
External models, checkpoints, and third-party code are under their respective licenses.

---

## Citation

If you use **LISTER** in your research, please cite:
```
Tao, Y. et al. (2025).
LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction.
GitHub: https://github.com/yutao3/LISTER
```

If you make use of specific integrated models or toolkits, please also cite the corresponding original works (DenseDepth, NeWCRFs, DepthFormer, diffusion-e2e-ft, NASA ASP, etc.).

---

## Contact

- Maintainer: **Yu Tao**  
- Email: **yu.tao@saiil.co.uk**  
- GitHub: **https://github.com/yutao3**

Contributions are welcome — please open issues or pull requests. For substantial changes, kindly discuss via an issue first.
