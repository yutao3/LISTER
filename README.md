# LISTER

**LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction**

LISTER is a research-oriented open-source toolbox for reconstructing lunar surface topography from LROC NAC images. It takes either a photogrammetric DTM or the LOLA DTM as reference, and provides an automated and modular framework for generating high-resolution digital terrain models (DTMs) using integrated pre-trained Monocular Depth Estimation networks and an end-to-end reconstruction pipeline.

---

## Features

- End-to-end automated pipelines for generating DTMs and surface reconstructions  
- Support for both **high-resolution photogrammetric DTMs** and **LOLA DTMs** as reference  
- Integration of dense u-net, vision transformers and diffusion-based monocular depth estimation models  
- Scripts for downloading, preprocessing, and converting raw LROC NAC images  
- Tools for brightness adjustment, quality filtering, georeferencing, mosaicking, and reprojection  
- Modular design allowing standalone use of helper scripts  

---

## Repository structure (key scripts)

- `LISTER_autoDTM.py` – main single-scale pipeline (photogrammetric DTM as reference)  
- `LISTER_autoDTM_MSP.py` – multi‑scale pyramid wrapper calling `LISTER_autoDTM.py` from coarse→fine  
- `prepare_geotiff_lroc_nac.py` – convert a raw LROC NAC EDR `.IMG` into an 8‑bit GeoTIFF via ISIS3 + GDAL  
- `pre_adjust_brightness.py` – shadow‑aware brightness/tone adjustment for GeoTIFFs  
- `check_low_quality_robust.py` – quality screening of GeoTIFFs in a folder  
- `get_latest_lroc_nac_list_and_merge.py` – build/refresh a master list of NAC EDR URLs (INDEX.TAB → one URL per line)  
- `get_lroc_nac.py` – download specific NAC EDRs using the master URL list  
- `compare2dtm.py` – ref/target DTM comparison across smoothing widths (RMSE & SSIM)  
- `quick_validation.py` – fast visual/statistical comparison of multiple DTMs over the common footprint  
- `cal_stat.py` – quick SSIM/RMSE between two images (utility)

> Note: `LISTER_autoDTM_MSP_light.py` is a simplified variant of the MSP wrapper and is intentionally not documented below.

---

## Installation

### 1) System dependencies

- **GDAL** (≥ 3.4) – command‑line tools and Python bindings  
- **NASA Ames Stereo Pipeline (ASP)** – used for **mosaicing** predicted tiles: `dem_mosaic`  
  - Repo: <https://github.com/NeoGeographyToolkit/StereoPipeline>
- **USGS ISIS3** – for processing raw LROC NAC EDRs into projected cubes/GeoTIFFs  
- **Python 3.9+** is recommended

Ensure `gdal_translate`, `gdalinfo`, `dem_mosaic`, and ISIS3 apps (e.g., `lronac2isis`, `lronaccal`, `spiceinit`, `lronacecho`, `cam2map`) are in your `PATH`.

### 2) Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If the `GDAL` wheel fails on your platform, install GDAL via your OS package manager first (ensuring the Python bindings match your interpreter), then `pip install rasterio` etc.

### 3) Pretrained weights

All network weights and the diffusion networks for **Marigold E2EFT** and **Stable Diffusion E2EFT** must be downloaded **separately** due to size limits. Use the Google Drive folder:

- **Weights & models:** <https://drive.google.com/drive/folders/1uQZtQKEiKxk3WJoZn2wMLVHRkegJQU6G?usp=drive_link>

Place the downloaded files under an accessible location and point `-w/--weights` (and any model‑specific paths) accordingly when running LISTER.

---

## Usage

### A. End-to-end pipeline (single-scale)

`LISTER_autoDTM.py` is the main script for running the pipeline with a reference **photogrammetric DTM**.

**CLI:**  
```
python LISTER_autoDTM.py -i INPUT.tif -r REF_DTM.tif -o OUTPUT_DTM.tif     -m MODEL_ID -w WEIGHTS.pth -a /path/to/ASP/bin [options]
```

**Arguments (exact, from the script):**  
- `-i, --input` **(required)**: Input **8‑bit 1‑channel GeoTIFF**.  
- `-r, --ref` **(required)**: Reference DTM GeoTIFF.  
- `-o, --output` **(required)**: Output DTM GeoTIFF path.  
- `-m, --model` **(required)**: Model ID for inference (e.g., `D`, `V`, `N`, …).  
- `-w, --weights` **(required)**: Pretrained model weights (`.pth`).  
- `-a, --asp` *(default: `~/Downloads/ASP/bin`)*: Path to ASP bin (expects `dem_mosaic`).  
- `-t, --tmp` *(default: `data_tmp`)*: Temporary working directory.

**Expert options:**  
- `--overlap` *(int, default **280**)*: Tile overlap (pixels).  
- `--valid_threshold` *(int, default **20**)*: Pixels ≥ threshold counted as valid (helps discard low‑texture/shadow).  
- `--max_nodata_pixels` *(int, default **3000**)*: Drop tiles still containing ≥ this many NoData pixels post‑inpaint.  
- `--ndv` *(float, default **-3.40282265508890445e+38**)*: NoData value for georeferenced tiles.  
- `--scale` *(float, default **2.75**)*: Gaussian scale factor used in post‑processing (LP/HP merge).  
- `--inpaint` *(flag)*: Enable inpainting for partially valid tiles.  
- `--inpaint_threshold` *(float, default **0.1**)*: Minimum valid‑pixel fraction to allow inpainting.  
- `--inpaint_method` `telea|ns` *(default **telea**)*: OpenCV inpainting algorithm.  
- `--fill_smoothing` *(int, default **1**)*: Smoothing iterations in FillNodata.

> Tip: The script accepts `--ndv -3.4e+38` or `--ndv=-3.4e+38`; both are handled internally.

**Example:**  
```bash
python LISTER_autoDTM.py   -i ./nac/PAIR_8BIT.tif   -r ./dtm/photogrammetric_ref.tif   -o ./out/PAIR_dtm.tif   -m V -w ./weights/ViT_model.pth   -a ~/miniconda3/envs/asp/bin   --overlap 280 --scale 2.75 --inpaint --inpaint_threshold 0.1
```

---

### B. LOLA‑referenced multi‑scale pipeline

`LISTER_autoDTM_MSP.py` wraps `LISTER_autoDTM.py` in a **coarse→fine** multi‑scale pyramid. Use this when the **LOLA DTM** is the reference, or whenever robust large‑area processing is desired. The wrapper automatically downsamples the input, forwards the flags to `LISTER_autoDTM.py`, and progressively tightens the post‑processing scale.

**CLI:**  
```
python LISTER_autoDTM_MSP.py -i INPUT.tif -r REF_DTM.tif -o OUTPUT_DTM.tif     -m MODEL_ID -w WEIGHTS.pth -a /path/to/ASP/bin [--num_of_scales N] [other expert flags]
```

**Arguments (exact, from the script):**  
- Inherited core flags: `-i/--input`, `-r/--ref`, `-o/--output`, `-m/--model`, `-w/--weights`, `-a/--asp`, `-t/--tmp`.  
- Expert flags forwarded to each level: `--overlap`, `--valid_threshold`, `--max_nodata_pixels`, `--ndv`, `--scale`, `--inpaint`, `--inpaint_threshold`, `--inpaint_method`, `--fill_smoothing`.  
- MSP specific: `--num_of_scales` *(int, default **3**)*: number of pyramid levels.

The wrapper chooses a coarsest level that is **not coarser than ~3× the reference DTM resolution** and evenly spaces scales up to full‑res. At level *k*, it uses `post_scale = (N - k) * base_scale`.

**Example:**  
```bash
python LISTER_autoDTM_MSP.py   -i ./nac/PAIR_8BIT.tif   -r ./dtm/LOLA_64ppd.tif   -o ./out/PAIR_dtm_lola_msp.tif   -m D -w ./weights/densedepth.pth   --num_of_scales 3 --overlap 280 --inpaint --inpaint_threshold 0.1
```

---

### C. Pre‑processing & utilities

#### 1) Convert LROC NAC EDR → 8‑bit GeoTIFF

`prepare_geotiff_lroc_nac.py` (ISIS3 + GDAL)  

**CLI:**  
```
python prepare_geotiff_lroc_nac.py input.IMG input_projection.map
```
This runs: `lronac2isis → spiceinit → lronaccal → lronacecho → cam2map → gdal_translate → 8‑bit scale`. Produces `input_8BIT.tif`.  (Intermediate `.cub/.tif` removed.)

#### 2) Brightness / tone adjustment (shadow‑aware)

`pre_adjust_brightness.py`  

**CLI:**  
```
python pre_adjust_brightness.py INPUT.tif OUTPUT.tif -m {clahe,agc,sigmoid,mask_gamma}   [--clip-limit 4.0] [--tile-size 64] [--gain 10.0] [--cutoff 0.5]   [--shadow-quantile 0.25] [--shadow-high-quantile 0.45] [--gamma 0.5] [--sigma 15.0]
```
Examples:
```bash
# Local shadow brightening
python pre_adjust_brightness.py in.tif out.tif -m mask_gamma   --shadow-quantile 0.25 --shadow-high-quantile 0.45 --gamma 0.45 --sigma 20

# Global sigmoid
python pre_adjust_brightness.py in.tif out_sigmoid.tif -m sigmoid --gain 8 --cutoff 0.45
```

#### 3) Quality screening of GeoTIFFs

`check_low_quality_robust.py`  

**CLI:**  
```
python check_low_quality_robust.py <input_directory> <output_text_file>
```
The script downsamples large images, computes four metrics (local STD, entropy, histogram spread, Laplacian mean), performs **adaptive** low/mid/high classification, and writes a tab‑separated report with per‑image metrics + averages.  

#### 4) Build a master list of NAC EDR download URLs

`get_latest_lroc_nac_list_and_merge.py` (standard library only)  

**CLI:**  
```
python get_latest_lroc_nac_list_and_merge.py OUTPUT_DIR [--mirror MIRROR_BASE]
```
Downloads available `INDEX/INDEX.TAB` files under each `LRO-L-LROC-2-EDR-V1.0/volume`, then merges the second column to one URL per line at `OUTPUT_DIR/all_lroc_nac_urls.txt`.  

#### 5) Download selected LROC NAC EDRs

`get_lroc_nac.py` (standard library only)  

**CLI:**  
```
python get_lroc_nac.py all_lroc_nac_urls.txt WANTED_IDS.txt OUTPUT_DIR
# or
python get_lroc_nac.py all_lroc_nac_urls.txt M1181811415LE OUTPUT_DIR
```
`WANTED_IDS.txt` contains one ID per line (with or without `.IMG`; `LE/RE` suffix inferred).  

#### 6) Compare two DTMs on a grid of smoothing widths

`compare2dtm.py`  

**CLI:**  
```
python compare2dtm.py REF_DTM.tif TAR_DTM.tif MAX_FILTER_WIDTH
```
Downsamples the reference to target resolution, crops to overlap, applies boxcar smoothing widths (up to `MAX_FILTER_WIDTH`), computes **RMSE** and **SSIM** per block, and saves `result.txt` and `result.png`.  

#### 7) Quick validation over overlap region (visual + stats)

`quick_validation.py`  

**CLI:**  
```
python quick_validation.py OUTPUT_DIR INPUT_IMAGE_8BIT.tif REF_DTM.tif DTM1.tif [DTM2.tif ...]
```
- Finds the **overlap** across all inputs; resamples all to the highest available resolution.  
- If the overlap is huge, takes a random 10–20% sub‑window for speed.  
- Produces side‑by‑side figures with scale bars and summary stats.  

#### 8) Simple SSIM/RMSE between two single‑band images

`cal_stat.py`  

**CLI:**  
```
python cal_stat.py IMAGE_A.tif IMAGE_B.tif
```
Prints SSIM and RMSE. (OpenCV/Scikit‑Image required.)  

---

## Data preparation tips

1. Convert raw NAC EDRs to projected **8‑bit** GeoTIFFs with `prepare_geotiff_lroc_nac.py`.  
2. Optionally run `pre_adjust_brightness.py` to lift shadows and improve texture before inference.  
3. Screen poor‑quality scenes using `check_low_quality_robust.py`.  
4. Ensure the reference DTM (photogrammetric or LOLA) **overlaps** your image AOI and has valid georeferencing.  
5. When using MSP, start with LOLA (global) or a coarse photogrammetric DTM and let the wrapper refine through scales.

---

## Models and third‑party components

LISTER integrates or supports pretrained monocular depth backbones:

- **DenseDepth (Dense U‑Net):** <https://github.com/ialhashim/DenseDepth>  
- **NeWCRFs (Vision Transformers):** <https://github.com/aliyun/NeWCRFs>  
- **DepthFormer (Vision Transformers):** <https://github.com/ashutosh1807/Depthformer>  
- **Diffusion E2E Finetuning (Marigold & Stable Diffusion):** base code derived from <https://github.com/VisualComputingInstitute/diffusion-e2e-ft>

Tiling mosaics use **NASA Ames Stereo Pipeline** (`dem_mosaic`): <https://github.com/NeoGeographyToolkit/StereoPipeline>.

> **Weights & checkpoints**: Download from the provided Google Drive (see above). Please ensure you comply with the original licenses of each upstream project when using or redistributing models and code.

---

## Acknowledgements

This work was carried out under a programme of and funded by the **European Space Agency (ESA)** LISTER project.  
We acknowledge the authors and maintainers of: DenseDepth, NeWCRFs, DepthFormer, the VCI Diffusion E2E‑FT repository (for Marigold and Stable Diffusion E2EFT), and the NASA Ames Stereo Pipeline.

---

## Citation

If you use LISTER in your research, please cite:

> Tao ***et al.***, **LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction**, 2025. (in preparation)

A BibTeX entry will be added when the paper/preprint is available.

---

## License

This repository is dual-licensed under:

Apache License 2.0 (a permissive open-source license), and
ESA Software Community License – Weak Copyleft (as required by Section 4.2 of the ESA contract).

You may choose to use this software under either license.
Both license texts are provided in the repository (LICENSE.Apache-2.0 and LICENSE.ESA-Community-Weak-Copyleft).
Third-party components retain their original licenses; please refer to the respective linked projects for details.

---

## Troubleshooting

- **`gdal`/`rasterio` import errors** → ensure system GDAL is installed and version‑matched with your Python interpreter.  
- **`dem_mosaic: command not found`** → add ASP `bin/` to `PATH` or pass `-a /path/to/ASP/bin`.  
- **Empty/black outputs** → try `pre_adjust_brightness.py -m mask_gamma` before inference; adjust `--valid_threshold`.  
- **Seams in mosaics** → increase `--overlap` and/or adjust `--scale`; try MSP wrapper for robustness.  
- **Large scenes** → use `LISTER_autoDTM_MSP.py` with `--num_of_scales 3` and sufficient disk in `--tmp`.

---

## Development notes

- Python 3.9+ recommended.  
- Scripts print progress and create intermediate artifacts under `--tmp`. These are cleaned upon success.  
- Contributions (issues/PRs) are welcome.
