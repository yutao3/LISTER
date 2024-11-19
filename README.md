# LISTER
A Pre-trained Open-Source Toolbox for Lunar Monocular Image to Surface Topography Estimation and Reconstruction

The LISTER toolbox is an open-source software framework designed to generate high-resolution digital elevation models (DEMs) of the lunar surface from single-view optical images. Leveraging state-of-the-art deep learning models, it automates the process from image input to DEM output, including image tiling, monocular depth estimation, georeferencing, and large-area mosaic creation. The toolbox supports integration with multiple lunar datasets, such as LROC and KAGUYA imagery, and includes validation tools to assess the accuracy and quality of outputs. Designed with scalability and modularity in mind, LISTER aims to serve planetary scientists and developers seeking efficient topographical analysis for lunar exploration and research.

# Acknowledgement
This work was carried out by Surrey AI Imaging Ltd funded by the European Space Agency under the programme of "Studies for Lunar Surface Software Development" (2024–2025).

We would like to extend our gratitude to the developers of three GitHub repositories that have been instrumental in the success of this project. Specifically, we have integrated network architectures into the LISTER toolbox from the following repositories:
1. NeWCRFs by aliyun: https://github.com/aliyun/NeWCRFs
2. Depthformer by ashutosh1807: https://github.com/ashutosh1807/Depthformer/tree/main
3. DenseDepth by ialhashim: https://github.com/ialhashim/DenseDepth
