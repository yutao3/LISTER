# Introduction

LISTER: A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction

The LISTER toolbox is an open-source software framework designed to generate high-resolution digital elevation models (DEMs) of the lunar surface from single-view optical images. Leveraging state-of-the-art deep learning models, it automates the process from image input to DEM output, including image tiling, monocular depth estimation, georeferencing, and large-area mosaic creation. The toolbox supports integration with multiple lunar datasets, such as LROC and KAGUYA imagery, and includes validation tools to assess the accuracy and quality of outputs. Designed with scalability and modularity in mind, LISTER aims to serve planetary scientists and developers seeking efficient topographical analysis for lunar exploration and research.

The pre-trained weights file needs to be downloaded separately via [SAIIL GoogleDrive](https://drive.google.com/drive/folders/1uQZtQKEiKxk3WJoZn2wMLVHRkegJQU6G?usp=sharing)
The NASA Ames Stereo Pipeline (ASP) is used for image and DEM mosaicing in the LISTER toolbox, and ASP can be accessed from [NASA Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline)

# Usage (Preliminary)

1. **Environment:**
   1.1. NVIDIA GPU is required (e.g., RTX 3090, RTX 4090)
   1.2. `pip install -r requirements.txt`
   1.3. Install NASA Ames stereo pipeline according to this guide: [NASA Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline)

2. **Important notes:**
   2.1. The input image specified following `--input` should be an 8-bit 1-channel geotiff image  
   2.2. The reference DTM specified following `--ref` should be a 32-bit 1-channel geotiff image  
   2.3. The input image and reference DTM should match pixel by pixel. Please verify in GIS software before initiating processing.  
   2.4. The output DTM should be specified following `--output`  
   2.5. The ASP `dem_mosaic` pipeline is used to mosaic the DTM tiles; please provide the ASP bin folder following `--asp`  
   2.6. The pre-trained model and weights can be specified following `--model` and `--weights`, respectively. Use `D` for DenseNet-161-U-Net, `V` for ViT-AB-U-Net, and `N` for NW-FC-CRF  
   2.7. A temporary working directory is required to store intermediate files. Ensure you have write permission for the space specified following `--tmp`

3. **Examples:**
   ```bash
   python full_chain.py --input data/input_image/xxx.tif --ref data/ref_dtm/yyy.tif --output data/output_dtm/zzz.tif --asp ~/Downloads/ASP/bin --model D --weights pre-trained-weights/mars-D.pth --tmp data/tmp_working_dir


# Acknowledgement

This work was carried out by Surrey AI Imaging Ltd funded by the European Space Agency under the programme of "Studies for Lunar Surface Software Development" (2024–2025).

We would like to extend our gratitude to the developers of three GitHub repositories that have been instrumental in the success of this project. Specifically, we have integrated network architectures into the LISTER toolbox from the following repositories:
[NeWCRFs by aliyun](https://github.com/aliyun/NeWCRFs)
[Depthformer by ashutosh1807](https://github.com/ashutosh1807/Depthformer/tree/main)
[DenseDepth by ialhashim](https://github.com/ialhashim/DenseDepth)
