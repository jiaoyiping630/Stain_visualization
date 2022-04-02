#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from pinglib_sep.utils import load_variables, save_variables
from pinglib_sep.monitor import terminal_viewer
from pinglib_sep.files import get_file_list, match_files, create_dir, purename
from pinglib_sep.imgprocessing import imread, imwrite
from pinglib_sep.color_clustering import color_palette, color_gmm, spectrum_in_xls, spectrum_distance, get_color_bar
from pinglib_sep.anchor_mds import anchored_mds_by_distance
from pinglib_sep.debugger import watch


#   Different options for spectrum distance
def image_color_manifold(anchor_image_paths,
                         float_image_paths=None,
                         anchor_masks=(1, -220, -220),  # list for paths, or tuple for threshold
                         float_masks=(1, -220, -220),
                         working_dir='',
                         component=5,
                         spectrum_method='palette',  # 'palette' or 'gmm'
                         spectrum_distance_method='assign',  # 'assign' for both method, 'js' for 'gmm'
                         pixel_distance=None,  # used for 'assign' mode, Euclidean for None
                         override=False):
    '''
        This function used for a quick analysis of staining condition in a cohort.
        Before running this script, you need to crop some patches from each slides.
        The paths of these image files will sent to  "anchor_image_paths"
        Result will be saved in "working_dir"

        It need to compute a nxn distance matrix between n slides, which might be costly.
        So there is another anchored version, which use anchor images to set up the coordinates,
        and locate float images by a mxn distance matrix.
        This can reduce (n+m)^2 -> nxn+nxm
        Anchored version will lose some information, so use normal version as first choice.
        When anchored version is used, make sure to select some representative image as anchor image.

    :param anchor_image_paths:
    :param float_image_paths:
    :param anchor_masks:
    :param float_masks:
    :param working_dir:
    :param component: cluster amount for K-means or GMM
    :param spectrum_method: method of descriptor, ['palette'|'gmm']
    :param spectrum_distance_method: method to measure distance between two image, ['assign'|js]
    :param pixel_distance: None for Euclidean distance, or any custom function defined as ((R1,G1,B1),(R2,G2,B2))
    :param override:
    :return:
    '''

    create_dir(working_dir)
    spectrum_dir = os.path.join(working_dir, 'spectrum')
    bar_dir = os.path.join(working_dir, 'bar')
    manifold_path = os.path.join(working_dir, 'manifold.pkl')
    manifold_mat_path = os.path.join(working_dir, 'manifold.mat')

    if float_image_paths is None:
        float_image_paths = []
        with_float_image = False
    else:
        with_float_image = True

    print('\n************** Step 1: Generate Spectrum ******************')

    anchor_image_amount = len(anchor_image_paths)
    float_image_amount = len(float_image_paths) if float_image_paths is not None else 0

    if not isinstance(anchor_masks, list):
        anchor_masks = [anchor_masks] * anchor_image_amount
    if not isinstance(float_masks, list):
        float_masks = [float_masks] * float_image_amount

    print('Generating spectrums for anchor images')
    for i, (image_path, mask) in enumerate(zip(anchor_image_paths, anchor_masks)):
        print('Processing ({}/{}) {} ...'.format(i + 1, anchor_image_amount, image_path))
        target_path = os.path.join(spectrum_dir, purename(image_path) + '.pkl')
        if not override and os.path.isfile(target_path):
            print('   target file exist, skipped')
            sys.stdout.flush()
            continue
        if isinstance(mask, str):
            mask = imread(mask)[0:2]
        if spectrum_method == 'palette':
            this_spectrum = color_palette(image_path, n=component, foreground=mask)
        elif spectrum_method == 'gmm':
            this_spectrum = color_gmm(image_path, n=component, foreground=mask)
        else:
            print('Unsupported method')
            raise NotImplementedError
        save_variables([this_spectrum], target_path)

    print('Generating spectrums for float images')
    for i, (image_path, mask) in enumerate(zip(float_image_paths, float_masks)):
        print('Processing ({}/{}) {} ...'.format(i + 1, float_image_amount, image_path))
        target_path = os.path.join(spectrum_dir, purename(image_path) + '.pkl')
        if not override and os.path.isfile(target_path):
            print('   target file exist, skipped')
            sys.stdout.flush()
            continue
        if isinstance(mask, str):
            mask = imread(mask)[0:2]
        if spectrum_method == 'palette':
            this_spectrum = color_palette(image_path, n=component, foreground=mask)
        elif spectrum_method == 'gmm':
            this_spectrum = color_gmm(image_path, n=component, foreground=mask)
        else:
            print('Unsupported method')
            raise NotImplementedError
        save_variables([this_spectrum], target_path)

    print('\n************** Step 2: Save Visual Results ******************')
    image_all = anchor_image_paths + float_image_paths
    spectrum_all = get_file_list(spectrum_dir, ext='pkl')
    [image_all, spectrum_all] = match_files([image_all, spectrum_all])
    color_bar_on_the_right(image_all, spectrum_all, bar_dir, override=override)

    spectrum_in_xls(spectrum_all, os.path.join(working_dir, 'spectrum.xlsx'))

    manifold = {}
    anchor_spectrums = get_file_list(spectrum_dir)
    [anchor_spectrums, anchor_image_paths] = match_files([anchor_spectrums, anchor_image_paths])
    if with_float_image:
        float_spectrums = get_file_list(spectrum_dir)
        [float_spectrums, float_image_paths] = match_files([float_spectrums, float_image_paths])
    else:
        float_spectrums = []
    anchor_purenames = [purename(path) for path in anchor_image_paths]
    float_purenames = [purename(path) for path in float_image_paths]
    manifold['anchor_image_paths'] = anchor_image_paths
    manifold['float_image_paths'] = float_image_paths
    manifold['anchor_image'] = anchor_purenames
    manifold['float_image'] = float_purenames

    print('\n************** Step 3: Generate Distance Matrix for Anchor Images *************')
    if os.path.isfile(manifold_path) and not override:
        [check_manifold] = load_variables(manifold_path)
        if 'anchor_to_anchor_distance' not in check_manifold.keys():
            need_compute = True
        else:
            need_compute = False
            anchor_to_anchor_distance = check_manifold['anchor_to_anchor_distance']
    else:
        need_compute = True
    if need_compute:

        anchor_to_anchor_distance = np.zeros((anchor_image_amount, anchor_image_amount))
        current = 0
        for i, pkl1 in enumerate(anchor_spectrums):
            for j, pkl2 in enumerate(anchor_spectrums):
                current += 1
                if i > j:
                    continue
                [spectrum1] = load_variables(pkl1)
                [spectrum2] = load_variables(pkl2)
                d = spectrum_distance(spectrum1, spectrum2, pixel_distance=pixel_distance,
                                      method=spectrum_distance_method)
                anchor_to_anchor_distance[i, j] = d
                anchor_to_anchor_distance[j, i] = d
                terminal_viewer(current, anchor_image_amount * anchor_image_amount)
        save_variables([manifold], target_path=manifold_path)
    manifold['anchor_to_anchor_distance'] = anchor_to_anchor_distance

    print('\n************** Step 4: Generate Distance Matrix between Floats and Anchors *************')
    if os.path.isfile(manifold_path) and not override and with_float_image:
        [check_manifold] = load_variables(manifold_path)
        if 'float_to_anchor_distance' not in check_manifold.keys():
            need_compute = True
        else:
            need_compute = False
            float_to_anchor_distance = check_manifold['float_to_anchor_distance']
    else:
        need_compute = True
    if need_compute:
        float_to_anchor_distance = np.zeros((float_image_amount, anchor_image_amount))

        current = 0
        for i, pkl1 in enumerate(float_spectrums):
            for j, pkl2 in enumerate(anchor_spectrums):
                current += 1
                [spectrum1] = load_variables(pkl1)
                [spectrum2] = load_variables(pkl2)
                d = spectrum_distance(spectrum1, spectrum2, pixel_distance=pixel_distance,
                                      method=spectrum_distance_method)
                float_to_anchor_distance[i, j] = d
                terminal_viewer(current, float_image_amount * anchor_image_amount)
        save_variables([manifold], target_path=manifold_path)
    manifold['float_to_anchor_distance'] = float_to_anchor_distance

    print('\n************** Step 5: Multi-dimensional Scaling *************')

    if os.path.isfile(manifold_path) and not override:
        [check_manifold] = load_variables(manifold_path)
        if ('anchor_coords' not in check_manifold.keys()) or ('float_coords' not in check_manifold.keys()):
            need_compute = True
        else:
            need_compute = False
            anchor_coords = check_manifold['anchor_coords']
            float_coords = check_manifold['float_coords']
    else:
        need_compute = True
    if need_compute:
        if with_float_image:
            [anchor_coords, float_coords] = anchored_mds_by_distance(anchor_to_anchor_distance,
                                                                     float_to_anchor_distance,
                                                                     dim=2)
        else:
            [anchor_coords, float_coords] = anchored_mds_by_distance(anchor_to_anchor_distance,
                                                                     None,
                                                                     dim=2)
    manifold['anchor_coords'] = anchor_coords
    manifold['float_coords'] = float_coords
    save_variables([manifold], target_path=manifold_path)

    print('\n************** Step 6: Save Results *************')
    anchor_spectrum_data = []
    for pkl_path in anchor_spectrums:
        [spectrum] = load_variables(pkl_path)
        anchor_spectrum_data.append([spectrum['weights'], spectrum['means'], spectrum['covariances'], ''])
    float_spectrum_data = []
    for pkl_path in float_spectrums:
        [spectrum] = load_variables(pkl_path)
        float_spectrum_data.append([spectrum['weights'], spectrum['means'], spectrum['covariances'], ''])
    manifold['anchor_spectrum'] = anchor_spectrum_data
    manifold['float_spectrum'] = float_spectrum_data
    save_variables([manifold], target_path=manifold_path)

    watch([manifold['anchor_image_paths'],
           manifold['float_image_paths'],
           manifold['anchor_image'],
           manifold['float_image'],
           manifold['anchor_to_anchor_distance'],
           manifold['float_to_anchor_distance'],
           manifold['anchor_coords'],
           manifold['float_coords'],
           manifold['anchor_spectrum'],
           manifold['float_spectrum']
           ],
          ['anchor_image_paths', 'float_image_paths',
           'anchor_image', 'float_image',
           'anchor_to_anchor_distance', 'float_to_anchor_distance',
           'anchor_coords', 'float_coords',
           'anchor_spectrum', 'float_spectrum'
           ],
          save_file_name=manifold_mat_path)


#   将聚类的色彩条组装到图像的右侧，方便观察
def color_bar_on_the_right(image_paths, spectrum_paths, save_dir='', override=False):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    image_amount = len(image_paths)
    for i, (image_path, clustering_path) in enumerate(zip(image_paths, spectrum_paths)):
        print('Processing ({}/{}) {} ...'.format(i + 1, image_amount, image_path))
        target_path = os.path.join(save_dir, purename(image_path) + '.jpg')
        if not override and os.path.isfile(target_path):
            print('   target file exist, skipped')
            sys.stdout.flush()
            continue
        image_data = imread(image_path)
        [spectrum] = load_variables(clustering_path)
        bar_width = int(image_data.shape[1] / 10)
        image_data_ext = np.zeros((image_data.shape[0], image_data.shape[1] + bar_width, 3), dtype='uint8')
        image_data_ext[0:image_data.shape[0], 0:image_data.shape[1], :] = image_data
        bar_data = get_color_bar(spectrum, bar_height=image_data.shape[0], bar_width=bar_width)
        image_data_ext[:, image_data.shape[1]:, :] = bar_data
        imwrite(image_data_ext, target_path)
