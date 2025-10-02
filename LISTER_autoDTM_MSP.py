#!/usr/bin/env python3
"""
LISTER_autoDTM_MSP.py: Multi-scale pyramid wrapper for LISTER_autoDTM.py
Implements coarse-to-fine multi-resolution processing by repeatedly calling
LISTER_autoDTM.py on downsampled versions of the input image.

Usage:
  python LISTER_autoDTM_MSP.py -i INPUT.tif -r REF.tif -o OUTPUT.tif \
       -m Model_ID -w WEIGHTS.pth [--num_of_scales N] [--inpaint] [other LISTER_autoDTM expert flags]

This script will:
 1) Optionally downsample large input images to multiple scales
 2) For each scale from coarse to full-resolution:
    - Resample input with gdal average
    - Call LISTER_autoDTM.py with the resampled input
      * Use reference DTM only at the coarsest level; thereafter use
        the output DTM from the previous (coarser) level
    - Adjust postprocessing scale factor: (num_of_scales - current_level) * base_scale
    - Store intermediate outputs as <output_base>_level{level}.tif
 3) If input smaller than 1000 px in width or height, runs a single LISTER_autoDTM call
 
HRD computes a data- and ref-aware starting scale: Ensures this coarsest scale is not coarser than ~3× the ref DTM resolution (uses GeoTransforms of both input and ref). It then linearly spaces scales between that coarse ratio and 1.0. 

"""
import os
import sys
import argparse
import subprocess
import time
from osgeo import gdal
import shutil  # [ADDED] for cleaning up tmp items created by this run

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-scale pyramid wrapper for LISTER_autoDTM.py"
    )
    # Inherit all arguments from LISTER_autoDTM.py
    parser.add_argument('-i', '--input', required=True,
                        help='Input 8-bit 1-channel GeoTIFF')
    parser.add_argument('-r', '--ref', required=True,
                        help='Reference DTM GeoTIFF')
    parser.add_argument('-o', '--output', required=True,
                        help='Output DTM GeoTIFF path')
    parser.add_argument('-m', '--model', required=True,
                        help='Model ID for inference (e.g., D, V, N, ...)')
    parser.add_argument('-w', '--weights', required=True,
                        help='Pretrained model weights (.pth)')
    parser.add_argument('-a', '--asp', default='~/Downloads/ASP/bin',
                        help='Path to ASP bin (dem_mosaic)')
    parser.add_argument('-t', '--tmp', default='data_tmp',
                        help='Temporary working directory')
    # Expert flags
    parser.add_argument('--overlap', type=int, default=280,
                        help='Tile overlap (px)')
    parser.add_argument('--valid_threshold', type=int, default=20,
                        help='Valid pixel intensity threshold')
    parser.add_argument('--max_nodata_pixels', type=int, default=3000,
                        help='Max nodata pixels allowed')
    parser.add_argument('--ndv', type=float,
                        default=-3.40282265508890445e+38,
                        help='NoData value for output tiles')
    parser.add_argument('--scale', type=float, default=2.75,
                        help='Guassian scale factor for postprocessing')
    parser.add_argument('--inpaint', action='store_true',
                        help='Enable image inpainting on near-valid tiles')
    parser.add_argument('--inpaint_threshold', type=float, default=0.1,
                        help='Min fraction valid pixels to inpaint')
    parser.add_argument('--inpaint_method', choices=['telea','ns'],
                        default='telea', help='Inpainting algorithm')
    parser.add_argument('--fill_smoothing', type=int, default=1,
                        help='Smoothing iterations in FillNodata')
    # MSP-specific
    parser.add_argument('--num_of_scales', type=int, default=3,
                        help='Number of scales in multi-scale pyramid')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def call_lister(script_path, inp, ref, out, tmp_dir, args, post_scale):
    """
    Call LISTER_autoDTM.py with forwarded parameters and adjusted scale.
    """
    cmd = [sys.executable, script_path,
           '-i', inp,
           '-r', ref,
           '-o', out,
           '-m', args.model,
           '-w', args.weights,
           '-a', args.asp,
           '--tmp', tmp_dir,
           '--overlap', str(args.overlap),
           '--valid_threshold', str(args.valid_threshold),
           '--max_nodata_pixels', str(args.max_nodata_pixels),
           '--ndv', str(args.ndv),
           '--scale', str(post_scale),
           '--fill_smoothing', str(args.fill_smoothing)]
    if args.inpaint:
        cmd.append('--inpaint')
        cmd.extend(['--inpaint_threshold', str(args.inpaint_threshold)])
        cmd.extend(['--inpaint_method', args.inpaint_method])
    print(f"[CALL] {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[ERROR] LISTER_autoDTM.py failed at scale with rc={ret.returncode}")
        sys.exit(ret.returncode)

def main():
    start_time = time.time()
    args = parse_args()
    args.asp = os.path.expanduser(args.asp)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    lister_script = os.path.join(script_dir, 'LISTER_autoDTM.py')

    # Ensure tmp exists; record snapshot so we can remove only new items later
    os.makedirs(args.tmp, exist_ok=True)
    try:
        pre_existing_tmp_items = set(os.listdir(args.tmp))
    except FileNotFoundError:
        pre_existing_tmp_items = set()

    # [ADDED] Ensure the parent directory for the output file exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        # Open datasets
        ds = gdal.Open(args.input)
        if ds is None:
            print(f"[ERROR] Cannot open input image: {args.input}")
            sys.exit(1)
        ds_ref = gdal.Open(args.ref)
        if ds_ref is None:
            print(f"[ERROR] Cannot open reference DTM: {args.ref}")
            sys.exit(1)

        # Geo info
        geot_in = ds.GetGeoTransform()
        geot_ref = ds_ref.GetGeoTransform()
        input_xres = abs(geot_in[1]); input_yres = abs(geot_in[5])
        ref_xres = abs(geot_ref[1]); ref_yres = abs(geot_ref[5])

        width, height = ds.RasterXSize, ds.RasterYSize

        # Small image: single-scale
        if width < 1000 or height < 1000:
            print("[INFO] Image is too small for multi-scale processing. Running single-scale.")
            single_tmp = os.path.join(args.tmp, 'single_scale')
            os.makedirs(single_tmp, exist_ok=True)
            call_lister(lister_script, args.input, args.ref,
                        args.output, single_tmp,
                        args, args.scale)
            print("[INFO] Single-scale processing complete.")
            total_elapsed = time.time() - start_time
            print(f"[TOTAL MSP TIME] {total_elapsed:.1f}s across 1 scale")
            print("[ALL DONE] Multi-scale processing complete.")
            return  # allow cleanup in finally

        # Compute initial coarse ratio by shape
        aspect = float(height) / float(width)
        ref_aspect = 480.0 / 640.0
        if aspect > ref_aspect:
            coarse_ratio = 640.0 / float(width)
        elif aspect < ref_aspect:
            coarse_ratio = 480.0 / float(height)
        else:
            coarse_ratio = 640.0 / float(width)

        # Ensure coarse resolution isn’t lower than reference DTM
        min_ratio = max(input_xres / ref_xres, input_yres / ref_yres)
        if coarse_ratio < 3 * min_ratio:
            print(f"[LEVEL 0] Adjusting coarse_ratio {coarse_ratio:.4f} -> {3*min_ratio:.4f} to match (3×ref) resolution")
            coarse_ratio = 3* min_ratio

        # Build evenly spaced scales between coarse and full-res
        N = args.num_of_scales
        if N < 2:
            scales = [1.0]
        else:
            scales = [coarse_ratio + float(i) * (1.0 - coarse_ratio) / float(N - 1)
                      for i in range(N)]

        base_out, ext = os.path.splitext(args.output)
        current_ref = args.ref

        for i, ds_ratio in enumerate(scales):
            post_scale = (N - i) * args.scale
            level_dir = os.path.join(args.tmp, f'level_{i}')
            os.makedirs(level_dir, exist_ok=True)
            lvl_input = os.path.join(level_dir, f'input_level{i}{ext}')

            if ds_ratio < 1.0:
                print(f"[LEVEL {i}] Downsampling (ratio {ds_ratio:.3f}) -> {lvl_input}")
                gdal.Translate(lvl_input, args.input,
                               format='GTiff',
                               width=int(width * ds_ratio),
                               height=int(height * ds_ratio),
                               resampleAlg=gdal.GRA_Average)
            else:
                print(f"[LEVEL {i}] Full-resolution copy -> {lvl_input}")
                gdal.Translate(lvl_input, args.input, format='GTiff')

            out_dtm = f"{base_out}_level{i}{ext}" if i < N-1 else args.output
            print(f"[LEVEL {i}] Calling LISTER_autoDTM with post-scale {post_scale:.3f}")
            call_lister(lister_script, lvl_input, current_ref,
                        out_dtm, level_dir, args, post_scale)
            current_ref = out_dtm

        total_elapsed = time.time() - start_time
        print(f"[TOTAL MSP TIME] {total_elapsed:.1f}s across {N} scales")
        print("[ALL DONE] Multi-scale processing complete.")

    finally:
        # [ADDED] Clean up only the items that were created during this run
        try:
            current_items = set(os.listdir(args.tmp))
        except FileNotFoundError:
            current_items = set()
        new_items = current_items - pre_existing_tmp_items
        for name in new_items:
            path = os.path.join(args.tmp, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
            except Exception as e:
                print(f"[WARN] Failed to remove tmp item '{path}': {e}")

if __name__ == '__main__':
    main()

