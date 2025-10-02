# LISTER

**LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction**

LISTER is a research-oriented open-source toolbox for reconstructing lunar surface topography from LROC NAC images. It integrates photogrammetric DTMs, LOLA data, and modern diffusion-based depth/normal estimation networks (Marigold E2E-FT and Stable Diffusion E2E-FT), providing an automated and modular framework for generating high-resolution digital terrain models (DTMs) and surface products.

---

## Features

- End-to-end automated pipelines for generating DTMs and surface reconstructions  
- Support for both **high-resolution photogrammetric DTMs** and **LOLA DTMs** as reference  
- Integration of diffusion-based monocular depth/normal estimation models  
- Scripts for downloading, preprocessing, and converting raw LROC NAC images  
- Tools for brightness adjustment, quality filtering, mosaicking, and reprojection  
- Modular design allowing standalone use of helper scripts  

---

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)  
   - [Automated Pipelines](#automated-pipelines)  
   - [Helper Scripts](#helper-scripts)  
4. [Models and Weights](#models-and-weights)  
5. [Outputs](#outputs)  
6. [Limitations](#limitations)  
7. [Acknowledgements](#acknowledgements)  
8. [License](#license)  
9. [Citation](#citation)  
10. [Contact](#contact)

---

## Requirements

- **Python**: 3.8 or newer (Python 3.10+ recommended)  
- **Core libraries**:  
  `numpy`, `scipy`, `Pillow`, `opencv-python`, `shapely`, `pyproj`, `affine`  
- **Geospatial**:  
  `gdal` (Python bindings) and GDAL command-line tools (`gdalwarp`, `gdal_translate`)  
  or `rasterio` (for some functions)  
- **Deep learning**:  
  `torch`, `torchvision`, `diffusers`, `transformers`  
- **Other**:  
  Internet access (for downloading LROC NAC unless you already have them), sufficient disk space (tens of GB for data + models), and a CUDA-capable GPU (strongly recommended for inference).  

---

## Installation

Clone this repository:

```bash
git clone https://github.com/yutao3/LISTER.git
cd LISTER
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` is not provided, manually install the packages listed above.)

Make sure GDAL is properly installed and available on your system:

```bash
gdalinfo --version
```

---

## Usage

### Automated Pipelines

The main entry points are:

- **With high-resolution photogrammetric DTM reference**  
  ```bash
  python LISTER_autoDTM.py --input-dir /path/to/nac_geotiffs --dtm /path/to/highres_dtm.tif
  ```

- **With LOLA DTM reference**  
  ```bash
  python LISTER_autoDTM_MSP.py --input-dir /path/to/nac_geotiffs --lola-dtm /path/to/lola.tif
  ```
  or
  ```bash
  python LISTER_autoDTM_MSP_light.py --input-dir ... --lola-dtm ...
  ```

Run `--help` on any script to see available options:

```bash
python LISTER_autoDTM.py --help
```

### Helper Scripts

| Script | Description |
|--------|-------------|
| `get_latest_lroc_nac_list_and_merge.py` | Fetches the latest LROC NAC image list and merges metadata |
| `get_lroc_nac.py` | Downloads LROC NAC images automatically given image IDs |
| `prepare_geotiff_lroc_nac.py` | Converts raw LROC NAC EDR images to GeoTIFF for processing |
| `check_low_quality_robust.py` | Detects and flags low-quality images |
| `pre_adjust_brightness.py` | Normalises brightness for consistent inputs |

These can be run standalone or used within the automated pipelines.

---

## Models and Weights

Due to file-size constraints, **pre-trained weights and diffusion networks are not included in this repository**.  

You must download the following separately from the [Google Drive link provided in the README](https://drive.google.com/) (see instructions in the repo):

- Marigold E2E-FT  
- Stable Diffusion E2E-FT  
- Any additional weight files required for inference  

After downloading, place them in a `models/` or `weights/` directory and ensure your scripts point to the correct paths.

Reference repository for diffusion models:  
[VisualComputingInstitute/diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)

---

## Outputs

Typical outputs include:

- High-resolution DTM (GeoTIFF)  
- Surface normal maps  
- Mosaics or stitched products aligned to reference DTMs  
- Quality/diagnostic maps and logs  
- Intermediate files (brightness-adjusted NACs, reprojections, etc.)  

---

## Limitations

- GPU recommended: inference with diffusion models is very slow on CPU  
- Alignment between NAC imagery and reference DTMs must be reasonable  
- Large datasets can consume significant disk space and memory  
- The "light" MSP pipeline trades resolution for speed/robustness  

---

## Acknowledgements

This repository integrates or builds upon:

- [VisualComputingInstitute/diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)  
  (Marigold E2E-FT and Stable Diffusion E2E-FT models)  
- Hugging Face implementations of diffusion depth/normal estimation models  
- GDAL, Rasterio, PyTorch, diffusers, transformers, and other open-source geospatial/ML libraries  

We gratefully acknowledge the authors of these projects for making their code and models publicly available.  

---

## License

This repository is released under the **MIT License** (unless otherwise stated in specific scripts).  

The diffusion models (Marigold E2E-FT and Stable Diffusion E2E-FT) are distributed under the **Apache 2.0 License**. Please review their individual licenses before use.

---

## Citation

If you use LISTER in your research, please cite:

```
Tao, Y. et al. (2025). 
LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction. 
GitHub: https://github.com/yutao3/LISTER
```

And also cite the diffusion networks if you make use of Marigold/Stable Diffusion E2E-FT:

> Martin Garcia, Gonzalo, et al.  
> *Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think*. WACV 2025.

---

## Contact

- Maintainer: Yu Tao  
- Email: yu.tao@saiil.co.uk  
- GitHub: [yutao3](https://github.com/yutao3)

Contributions are welcome — please open issues or pull requests.
