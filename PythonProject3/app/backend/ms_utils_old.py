# This file contains functions to analyze multispectral pathology images.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from joblib import Parallel, delayed

# for thresholding
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_erosion, disk, remove_small_objects
from skimage.measure import label, regionprops

# for finding the bacteria centers using watershed segmentation
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

import scipy.ndimage as nd

from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import seaborn as sns
import random

from collections import deque

from datasets import Dataset, DatasetDict, load_from_disk, Features, Value, Array2D

import spectral # pip install spectral, a comprehensive toolkit for working with hyperspectral imagery


# 设置字体为文泉驿正黑
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


def example_usage_visualization_and_PCA():
    """
    Example usage of the functions.
    """
    # Example usage
    envi_file = '/mnt/truenas_datasets/Multispectral_Pathology/省人民医院/志贺菌/宋内志贺菌-75%酒精灭菌-标本编号-001/1-宋内志贺菌.HDR'  # for both HDR and raw files
    bmp_file    = '/mnt/truenas_datasets/Multispectral_Pathology/省人民医院/志贺菌/宋内志贺菌-75%酒精灭菌-标本编号-001/1-宋内志贺菌.bmp'
    bmp_image = mpimg.imread(bmp_file)

    # Show preview image (using the preset RGB bands)
    device_id = '001'
    RGB_band_indexes = find_device_RGB_band_indexes(device_id, header)
    print(RGB_band_indexes)
    # spectral.imshow(ms_data, bands=band_indexes)  # only work in ipynb (ipython)

    # Simulate the RGB image from the multispectral data
    rgb_simulated = simulate_rgb_from_spectrum(device_id, header, ms_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    aligned_bmp_image = np.flipud(np.rot90(bmp_image))
    axes[0].imshow(aligned_bmp_image)
    axes[0].set_title('BMP Image')

    # Display the RGB image of the multispectral data
    axes[1].imshow(rgb_simulated)
    axes[1].set_title('Multispectral RGB Image')

    # Adjust the layout and display the images
    plt.tight_layout()
    plt.show(block=True)

    # PCA on the multispectral data
    pc_object = spectral.principal_components(ms_data)  # retruns a PrincipalComponents object

    # 保留99%的方差
    pc_object_reduced = pc_object.reduce(fraction=0.99)

    ms_data_reduced = pc_object_reduced.transform(ms_data)  # centered data
    print(ms_data_reduced.shape)
    # 降维后的数据可视化
    spectral.imshow(ms_data_reduced, bands=[0,1,2])
    plt.show(block=True)

    # view = spectral.view_nd(ms_data_reduced)  


def example_usage_bacteria_segmentation():
    # Example usage for bacteria extraction
    envi_file = '/mnt/truenas_datasets/Multispectral_Pathology/省人民医院/志贺菌/宋内志贺菌-75%酒精灭菌-标本编号-001/1-宋内志贺菌.HDR'  # for both HDR and raw files
    bmp_file    = '/mnt/truenas_datasets/Multispectral_Pathology/省人民医院/志贺菌/宋内志贺菌-75%酒精灭菌-标本编号-001/1-宋内志贺菌.bmp'
    bmp_image = mpimg.imread(bmp_file)

    # Load the image and header
    ms_data, header = load_ms_image(envi_file)
    print(header)

    # Convert the multispectral image to grayscale
    gray_image = convert_ms_to_gray(ms_data)

    # Apply binary thresholding
    mask = binary_thresholding(gray_image)

    # Find the bacteria centers
    centers = find_bacteria_centers(mask, gray_image, max_area=500, min_distance=10, exclude_border=5)
    print(f'Number of bacteria centers found: {len(centers)}')

    show_bacteria_centers(gray_image, mask, centers)

    # If you only want to handle 100 bacteria, sample them (if we have more)
    np.random.shuffle(centers)
    chosen_centers = centers[:150]  # or fewer if centers < 100

    # Extract up to 100 connected pixels for each center
    all_coords = extract_bacteria(mask, centers, max_pixels=50)
    
    # Visualize all extracted bacteria in one figure
    visualize_extraction_all(gray_image, mask, all_coords, 
                             show_mask=True, 
                             title="All Extracted Bacteria")
BAND_START = 0  # 波段起始位置
BAND_END = 257  # 波段终止位置
# NUM_BANDS = BAND_END - BAND_START + 1  # 数据中有258和260两种波段数，段波长和长波长噪声太大，舍去
NUM_BANDS = 258

# Preset wavelengths for each device
preset_wavelengths = {
    '001': {'R': 573, 'G': 527, 'B': 482},
    '007': {'R': 575, 'G': 526, 'B': 478},
    '008': {'R': 563, 'G': 520, 'B': 495},
}

def find_closest_wavelength_index(wavelengths, target_wavelength):
    """
    Find the index of the wavelength in the wavelengths list that is closest to the target_wavelength.
    """
    differences = np.abs(np.array(wavelengths) - target_wavelength)
    return np.argmin(differences)

def find_device_RGB_band_indexes(device_id, header):
    """
    Given a device_id and header file with wavelengths, find the indices of the R, G, and B bands for that device.
    """
    device_RGB = preset_wavelengths.get(device_id)
    device_wavelengths = [float(x) for x in header['wavelength']]

    # Find the indices for R, G, and B wavelengths
    r_index = find_closest_wavelength_index(device_wavelengths, device_RGB['R'])
    g_index = find_closest_wavelength_index(device_wavelengths, device_RGB['G'])
    b_index = find_closest_wavelength_index(device_wavelengths, device_RGB['B'])

    return [r_index, g_index, b_index]


def data_normalization(data_array, method='channelwise'):
    data_norm = np.zeros_like(data_array)

    if method == 'channelwise': # Normalize each channel independently
        for c in range(data_array.shape[-1]):
            channel = data_array[:, :, c]
            c_min = channel.min()
            c_max = channel.max()
            # Avoid division by zero by adding a small epsilon
            data_norm[:, :, c] = (channel - c_min) / (c_max - c_min + 1e-8)
    elif method == 'global':
        data_norm = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array) + 1e-8)
    else:
        raise ValueError('Invalid normalization method')

    return data_norm


def visualize_pca(data, n_components=3):
    """
    Apply PCA on the multi-band image and visualize the first 3 components as RGB.
    """
    # Reshape the data into (lines * samples, bands)
    reshaped_data = data.reshape(-1, data.shape[2])

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)

    # Reshape the PCA result back into the image format (lines, samples, n_components)
    pca_image = pca_result.reshape(data.shape[0], data.shape[1], n_components)

    # Plot the first 3 PCA components as RGB
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(pca_image / np.max(pca_image), aspect='auto')  # Normalize the values
    ax.set_title("PCA Components")
    plt.show()


def visualize_wavelengths(data, bands_to_show=[10, 30, 50]):
    """
    Visualize specific bands from the hyperspectral data (e.g., bands 10, 30, 50).
    These bands are shown as RGB images.
    """
    # Select bands
    selected_bands = data[:, :, bands_to_show]
    
    # Normalize the data for display
    selected_bands = (selected_bands - np.min(selected_bands)) / (np.max(selected_bands) - np.min(selected_bands))

    # Plot the selected bands as RGB
    plt.imshow(selected_bands)
    plt.title(f"Selected Bands {bands_to_show}")
    plt.show()

def visualize_rgb_from_device(device_id, header, raw_data):
    """
    Given a device_id, header file with wavelengths, and raw image data, 
    extract the R, G, B bands and show them as an RGB image.
    """
    # Get the preset wavelengths for the device
    device_wavelengths = preset_wavelengths.get(device_id)
    if not device_wavelengths:
        raise ValueError(f"Device ID '{device_id}' not found in preset wavelengths.")
    
    # Extract the wavelength list from the header
    wavelengths = header['wavelength']
    
    # Find the indices for R, G, and B wavelengths
    r_index = find_closest_wavelength_index(wavelengths, device_wavelengths['R'])
    g_index = find_closest_wavelength_index(wavelengths, device_wavelengths['G'])
    b_index = find_closest_wavelength_index(wavelengths, device_wavelengths['B'])
    
    # Extract the R, G, B bands from the raw data
    r_band = raw_data[:, :, r_index]
    g_band = raw_data[:, :, g_index]
    b_band = raw_data[:, :, b_index]
    
    # Normalize the bands for display
    rgb_image = np.zeros((raw_data.shape[0], raw_data.shape[1], 3))
    rgb_image[:, :, 0] = (r_band - np.min(r_band)) / (np.max(r_band) - np.min(r_band))  # Red
    rgb_image[:, :, 1] = (g_band - np.min(g_band)) / (np.max(g_band) - np.min(g_band))  # Green
    rgb_image[:, :, 2] = (b_band - np.min(b_band)) / (np.max(b_band) - np.min(b_band))  # Blue
    
    # Display the RGB image
    plt.imshow(rgb_image)
    plt.title(f"True Color Image for Device {device_id}")
    plt.show()


# Function to extract wavelengths from the header
def extract_wavelengths_from_header(header):
    """
    Extract wavelengths from the header file and convert to a list of floats.
    """
    return np.array([float(x) for x in header['wavelength']])

# Function to calculate RGB response curves based on wavelengths
def calculate_rgb_response(wavelengths, RGB_Center):
    """
    Calculate the RGB response curves for a typical color camera based on the given wavelengths.
    """
    # Define the typical RGB spectral response functions (example Gaussian functions)
    # These are for illustrative purposes and can be replaced with actual response data if available.

    # Red channel response (approximate, centered around 650 nm)
    R_response = np.exp(-0.5 * ((wavelengths - RGB_Center['R']) / 42.5)**2)

    # Green channel response (approximate, centered around 550 nm)
    G_response = np.exp(-0.5 * ((wavelengths - RGB_Center['G']) / 46.7)**2)

    # Blue channel response (approximate, centered around 450 nm)
    B_response = np.exp(-0.5 * ((wavelengths - RGB_Center['B']) / 38.2)**2)

    # Normalize the responses so that they sum to 1 (to keep the relative intensity)
    R_response /= np.sum(R_response)
    G_response /= np.sum(G_response)
    B_response /= np.sum(B_response)

    return R_response, G_response, B_response

# Function to simulate RGB image from multispectral data
def simulate_rgb_from_spectrum(device_id, header, ms_data):
    """
    Simulate the RGB image from the multispectral data using the RGB response functions.
    """
    # Convert wavelengths to float
    device_wavelengths = extract_wavelengths_from_header(header)

    # Calculate the RGB response curves
    RGB_Center = preset_wavelengths.get(device_id)
    R_response, G_response, B_response = calculate_rgb_response(device_wavelengths, RGB_Center)

    # Extract the corresponding bands for each RGB channel
    R_band = np.sum(ms_data * R_response, axis=-1)  # Weighted sum for Red channel
    G_band = np.sum(ms_data * G_response, axis=-1)  # Weighted sum for Green channel
    B_band = np.sum(ms_data * B_response, axis=-1)  # Weighted sum for Blue channel

    R_band = (R_band - np.min(R_band)) / (np.max(R_band) - np.min(R_band))  # Normalize to [0, 1]
    G_band = (G_band - np.min(G_band)) / (np.max(G_band) - np.min(G_band))  # Normalize to [0, 1]
    B_band = (B_band - np.min(B_band)) / (np.max(B_band) - np.min(B_band))  # Normalize to [0, 1]

    # Stack the channels back into a single RGB image
    rgb_image = np.dstack([R_band, G_band, B_band])

    # Normalize to [0, 255] for visualization
    rgb_image = np.clip(rgb_image, 0, 1)  # Ensure the values are within [0, 1]
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    return rgb_image


def load_ms_image(envi_file):
    """
    Load the multispectral image and header file.
    """
    img = spectral.open_image(envi_file)

    # Access metadata
    header = img.metadata
    if header['interleave'] == 'bsq':
        print(f'Warning: interleave is bsq, cannot load data: {envi_file}')
        return None, None

    # Access the image data (numpy array)
    ms_data = img.load()  # This loads the raw data into memory

    # 前2-3列经常出现异常值，保险起见舍去5列
    ms_data = ms_data[:, 5:, :]  # Remove the first 5 columns (high noise)
    header['samples'] = ms_data.shape[1]

    # Apply Gaussian smoothing only along the spectral dimension (axis=2)
    # sigma = 1  # adjust sigma as needed
    # ms_data = nd.gaussian_filter1d(ms_data, sigma=sigma, axis=2)

    # # Only use bands BAND_START to BAND_END (inclusive)
    # ms_data = ms_data[:, :, BAND_START:BAND_END+1] 
    # # Modify the header to reflect the new band count
    # header['bands'] = BAND_END - BAND_START + 1
    # header['wavelength'] = header['wavelength'][BAND_START:BAND_END+1]
    
    return ms_data, header


def convert_ms_to_gray(ms_data):
    """
    Convert the multispectral image to grayscale by averaging the bands.
    """
    gray_image = np.mean(ms_data, axis=-1)  # Average the bands to get grayscale image
    gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image) + 1e-8)  # Normalize to [0, 1]
    return gray_image


def binary_thresholding(gray_image):
    """
    Apply binary thresholding (OTSU method) to the grayscale image.
    """
    # Compute Otsu threshold.
    # threshold_value = threshold_otsu(gray_image)
    threshold_value = threshold_local(gray_image, block_size=51, method='gaussian', offset=0.01)

    # Create a binary mask.
    binary_mask = gray_image < threshold_value

    # Erode (shrink) the mask using a disk structuring element.
    # Increase or decrease 'disk(2)' to change how much the mask is shrunk.
    eroded_mask = binary_erosion(binary_mask, footprint=disk(1))

    # Remove small isolated regions from the eroded mask.
    # Adjust 'min_size' based on your application; e.g., 50, 100, 200, etc.
    cleaned_mask = remove_small_objects(eroded_mask, min_size=50)

    # Remove stripe-like artifacts from the cleaned mask.
    labeled_mask = label(cleaned_mask)
    props = regionprops(labeled_mask)
    for r in props:
        # Example: if the major axis is much bigger than the minor axis, 
        # or eccentricity is too high, remove it.
        if r.minor_axis_length < 1e-6:
            labeled_mask[labeled_mask == r.label] = 0
            continue

        if r.major_axis_length / r.minor_axis_length > 5:  # adjust threshold
            labeled_mask[labeled_mask == r.label] = 0

    final_mask = labeled_mask > 0

    return final_mask


def find_bacteria_centers_v1(mask, max_area=500, min_distance=10, exclude_border=5):
    """
    Identify individual bacteria centers by:
      1) Labeling connected components in `mask`.
      2) For each large region (> max_area), run a watershed segmentation
         based on the distance transform to split it.
      3) Compute and return the centroid for each final subregion.

    Parameters
    ----------
    mask : 2D array-like
        Binary mask (True/1 for bacteria, False/0 for background).
    max_area : int
        The maximum area above which a connected region is considered "large"
        and will be split via watershed.
    min_distance : int
        Minimum distance (in pixels) between local maxima in the distance transform,
        used for peak_local_max. Larger values produce fewer splits.

    Returns
    -------
    centers : list of (float, float)
        List of (row, column) centroids, one per final subregion.
    """

    # Label all connected regions
    labeled = label(mask)
    regions = regionprops(labeled)

    all_centers = []

    for region in regions:
        # If region is small enough, just take its centroid
        if region.area <= max_area:
            all_centers.append(region.centroid)
        else:
            # Split this large region using watershed
            # 1) Extract the bounding box sub-mask for this region
            minr, minc, maxr, maxc = region.bbox
            submask = (labeled[minr:maxr, minc:maxc] == region.label)

            # 2) Compute distance transform within this subregion
            distance_map = distance_transform_edt(submask)

            # 3) Find local maxima coordinates (returns Nx2 array of (row, col))
            coords = peak_local_max(
                distance_map,
                min_distance=min_distance,
                exclude_border=exclude_border,
                labels=submask
            )

            # Convert the coordinates into a boolean mask
            local_max = np.zeros_like(distance_map, dtype=bool)
            local_max[tuple(coords.T)] = True

            # 4) Label the peaks to get unique markers
            markers = label(local_max)

            # 4) Run the watershed on the NEGATIVE distance map
            #    (so that high distance = deep "basin").
            #    mask=submask ensures we only segment within this region.
            w_labels = watershed(-distance_map, markers, mask=submask)

            # 5) Measure each watershed subregion and compute centroids
            w_props = regionprops(w_labels)
            for wreg in w_props:
                # Convert subregion centroid to absolute coordinates
                sub_centroid = wreg.centroid
                abs_centroid = (
                    sub_centroid[0] + minr,
                    sub_centroid[1] + minc
                )
                all_centers.append(abs_centroid)

    return np.array(all_centers).astype(int)


def find_bacteria_centers_v2(mask,
                          gray,
                          max_area=500,
                          min_distance=10,
                          exclude_border=5):
    """
    Identify individual bacteria centers by:
      1) Labeling connected components in `mask`.
      2) For each large region (> max_area), run a watershed segmentation 
         on the inverted grayscale to split it into darker "basins."
      3) Compute and return the centroids of each subregion.

    Parameters
    ----------
    mask : 2D np.ndarray of bool
        Binary mask (True/1 = bacteria, False/0 = background).
    gray : 2D np.ndarray (float or uint8)
        Grayscale image in the same shape as 'mask'. 
        Darker intensities = darker bacteria.
    max_area : int
        The maximum area above which a connected region is considered "large"
        and will be split via watershed.
    min_distance : int
        Minimum distance (in pixels) between local maxima in the inverted
        grayscale. Larger values produce fewer splits.
    exclude_border : int
        Exclude local maxima found within 'exclude_border' pixels of the subregion
        boundary.

    Returns
    -------
    centers : np.ndarray of shape (N, 2), dtype=int
        Array of (row, column) centroids, one per final subregion.
    """
    # Label all connected regions in the mask
    labeled = label(mask)
    regions = regionprops(labeled)

    all_centers = []

    for region in regions:
        # For small or moderately sized regions, just use the region centroid
        if region.area <= max_area:
            all_centers.append(region.centroid)
            continue

        # For large regions, do a local watershed using grayscale info
        minr, minc, maxr, maxc = region.bbox
        submask = (labeled[minr:maxr, minc:maxc] == region.label)
        
        # Extract the corresponding grayscale subregion
        gray_sub = gray[minr:maxr, minc:maxc]

        # 1) Invert the grayscale subregion so that dark becomes 'high'
        #    i.e. local maxima in dark_map correspond to dark minima in gray_sub
        dark_map = gray_sub.max() - gray_sub

        # 2) Find local maxima in dark_map
        coords = peak_local_max(
            dark_map,
            min_distance=min_distance,
            exclude_border=exclude_border,
            labels=submask
        )

        # Create a boolean mask of local maxima
        local_max = np.zeros_like(dark_map, dtype=bool)
        if len(coords) > 0:
            local_max[tuple(coords.T)] = True

        # 3) Label these peaks to get watershed markers
        markers = label(local_max)

        # 4) Watershed on the negative grayscale 
        #    (so darkest regions become "deep basins")
        w_labels = watershed(-gray_sub, markers, mask=submask)

        # 5) Measure each watershed subregion and compute centroids
        w_props = regionprops(w_labels)
        for wreg in w_props:
            sub_centroid = wreg.centroid  # local coords in subregion
            abs_centroid = (
                sub_centroid[0] + minr,
                sub_centroid[1] + minc
            )
            all_centers.append(abs_centroid)

    # Convert to integer for use in BFS or pixel indexing
    return np.array(all_centers).astype(int)


def find_bacteria_centers(
    mask,
    gray,
    max_area=500,
    min_distance=3,      # smaller default
    exclude_border=1     # smaller default
):
    """
    Identify individual bacteria centers by labeling connected components 
    in `mask` and for large regions, using a grayscale-based watershed on 
    locally normalized subregions (dark = interior).
    """
    labeled = label(mask)
    regions = regionprops(labeled)
    all_centers = []

    for region in regions:
        # For small regions, just assume single bacterium
        if region.area <= max_area:
            all_centers.append(region.centroid)
            continue

        # Large region => local watershed
        minr, minc, maxr, maxc = region.bbox
        submask = (labeled[minr:maxr, minc:maxc] == region.label)
        
        # Extract local grayscale
        gray_sub = gray[minr:maxr, minc:maxc]
        
        # Local normalization
        local_min, local_max = gray_sub.min(), gray_sub.max()
        denom = (local_max - local_min) if (local_max > local_min) else 1e-8
        gray_sub_norm = (gray_sub - local_min) / denom
        
        # Invert so dark => high
        dark_map = 1.0 - gray_sub_norm
        
        # Find local maxima (darkest spots in original)
        coords = peak_local_max(
            dark_map,
            min_distance=min_distance,
            exclude_border=exclude_border,
            labels=submask
        )
        
        # Create boolean mask of local maxima
        local_maxima_mask = np.zeros_like(dark_map, dtype=bool)
        if len(coords) > 0:
            local_maxima_mask[tuple(coords.T)] = True
        
        # Label markers
        markers = label(local_maxima_mask)
        
        # Watershed on negative (back to "dark is deep")
        w_labels = watershed(-gray_sub_norm, markers, mask=submask)

        # Compute subregion centroids
        w_props = regionprops(w_labels)
        for wreg in w_props:
            s_cy, s_cx = wreg.centroid
            # Convert local coords to global
            abs_centroid = (s_cy + minr, s_cx + minc)
            all_centers.append(abs_centroid)

    return np.array(all_centers).astype(int)


def show_bacteria_centers(gray_image, mask, centers):
    """
    Show the bacteria centers on the image.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(gray_image, cmap='gray')
    ax[0].set_title('Grayscale Image')

    masked_gray = np.zeros_like(gray_image)
    masked_gray[mask] = gray_image[mask]
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Masked Image')

    ax[2].imshow(masked_gray, cmap='gray')
    # ax[2].plot([c[1] for c in centers], [c[0] for c in centers], 'ro')

    # If there are no centers, just show the mask
    if len(centers) > 0:
        # Overlay red scatter points at each (col, row), 
        # since plt.scatter expects x=column, y=row
        plt.scatter(
            centers[:, 1],  # x
            centers[:, 0],  # y
            c='red', 
            s=1, 
            marker='.'
        )

    ax[2].set_title('Bacteria Centers')
    
    plt.show()


def extract_bacteria(mask, centers, max_pixels=50):
    """
    For each center in 'centers', gather up to 'max_pixels' connected pixels
    from 'mask' via BFS. Returns a list of lists of pixel coordinates.
    
    Parameters
    ----------
    mask : 2D np.ndarray of bool
        Binary mask (True = bacterium, False = background).
    centers : np.ndarray or list of shape (N, 2)
        Each row is (row_center, col_center) in int.
    max_pixels : int
        The maximum number of pixels to collect for each bacterium.
    
    Returns
    -------
    all_bacteria_coords : list of list of (int, int)
        A list (one entry per center) of pixel coordinates (row, col).
        If the region is smaller than 'max_pixels', the entire region is returned.
    """
    rows, cols = mask.shape
    all_bacteria_coords = []

    # 8-neighbor offsets (can switch to 4 if desired)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    for center in centers:
        # Ensure center is integer
        # center could be an array or tuple, e.g. [row, col]
        row_center, col_center = map(int, center)

        # Validate bounds and mask
        if not (0 <= row_center < rows and 0 <= col_center < cols):
            continue
        if not mask[row_center, col_center]:
            continue
        
        visited = set()
        visited.add((row_center, col_center))
        queue = deque([(row_center, col_center)])
        
        coords = []
        
        while queue and len(coords) < max_pixels:
            r, c = queue.popleft()
            coords.append((r, c))
            
            # Explore neighbors
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    if mask[rr, cc] and (rr, cc) not in visited:
                        visited.add((rr, cc))
                        queue.append((rr, cc))
        
        # Only keep this bacterium if we have enough pixels
        if len(coords) == max_pixels:
            all_bacteria_coords.append(coords)

    return all_bacteria_coords


def visualize_extraction_all(gray_image, mask, all_bacteria_coords, 
                             show_mask=True, title="Extracted Bacteria"):
    """
    Show the 2D 'gray_image' and optionally the 'mask' as a semi-transparent overlay.
    Then plot all extracted bacterium pixels (each set in all_bacteria_coords) in one figure.
    
    Parameters
    ----------
    gray_image : 2D np.ndarray
        A grayscale version of the image (for background).
    mask : 2D np.ndarray of bool
        The binary mask (True = bacterium, False = background).
    all_bacteria_coords : list of list of (int, int)
        A list of lists; each sub-list contains pixel coords (row, col) 
        for a single extracted bacterium region.
    show_mask : bool
        If True, overlay the entire mask in a semi-transparent color. 
    title : str
        Title for the display.
    """
    plt.figure(figsize=(10, 8))
    
    # 1) Display the grayscale image
    masked_gray = np.zeros_like(gray_image)
    masked_gray[mask] = gray_image[mask]
    plt.imshow(masked_gray, cmap='gray')
    
    # 2) (Optional) Show the entire mask with light transparency
    # if show_mask:
    #     overlay = mask.astype(float)
    #     plt.imshow(overlay, cmap='spring', alpha=0.2)
    
    # 3) Plot each bacterium's pixels in a different color
    #    We'll pick from a color map or random colors
    #    For simplicity, let's use a single colormap with a fixed number of colors
    # cmap = plt.cm.get_cmap('rainbow', len(all_bacteria_coords))
    
    for i, coords in enumerate(all_bacteria_coords):
        if len(coords) == 0:
            continue
        coords_array = np.array(coords)
        # Scatter them, using a distinct color for each bacterium
        plt.scatter(coords_array[:, 1], coords_array[:, 0],
                    color='red',
                    s=1, marker='.')
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def sample_mask_pixels(mask, ms_data=None, num_samples=500, random_state=None):
    """
    Uniformly sample 'num_samples' pixels from the given 'mask' (True=foreground).
    If 'ms_data' is provided, also return the associated spectral vectors.
    
    Parameters
    ----------
    mask : 2D np.ndarray of bool
        Cleaned binary mask (True = bacterium region, False = background).
    ms_data : 3D np.ndarray of shape (H, W, num_bands) or None
        If provided, we extract the spectral data of the sampled pixels.
        Must match mask shape on the first two dimensions.
    num_samples : int
        Number of pixels to sample. If mask has fewer than num_samples
        pixels, return all of them.
    random_state : int or None
        Seed for reproducible sampling.
    
    Returns
    -------
    coords : np.ndarray of shape (k, 2)
        The (row, col) coordinates of sampled pixels. k = min(num_samples, #mask_pixels).
    spectra : np.ndarray of shape (k, num_bands) or None
        The spectral vectors for each sampled pixel (if ms_data is provided).
        Otherwise returns None.
    """
    # Optionally set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # 1) Find all foreground pixel coordinates in the mask
    all_coords = np.argwhere(mask)  # shape: (N, 2)

    num_mask_pixels = len(all_coords)
    if num_mask_pixels == 0:
        # No pixels in mask, return empty
        return np.empty((0, 2), dtype=int), None if ms_data is not None else None

    # 2) Decide how many to sample
    if num_mask_pixels <= num_samples:
        sample_indices = np.arange(num_mask_pixels)
    else:
        # Randomly choose 'num_samples' distinct indices
        sample_indices = np.random.choice(num_mask_pixels, size=num_samples, replace=False)
    
    coords = all_coords[sample_indices]

    # 3) If ms_data is provided, collect spectral vectors
    spectra = None
    if ms_data is not None:
        # ms_data shape: (H, W, num_bands)
        # coords shape: (k, 2) => (row, col)
        # Gather spectral data for each (r, c)
        spectral_list = []
        for (r, c) in coords:
            spectral_list.append(ms_data[r, c, :])
        spectra = np.vstack(spectral_list)  # shape = (k, num_bands)
    
    spectra = np.array(spectra).squeeze()

    if spectra is None:
        return None, None

    if spectra.shape[1] == 260: # 如果是260维数据，则去掉前后的2个维度
        spectra = spectra[:, 1:-1]
    assert spectra.shape[1] == NUM_BANDS, "Number of bands error!"

    return coords, spectra


def sample_mask_pixels_block_avg(mask, ms_data, num_blocks=5):
    """
    Divide the image (mask) into 'num_blocks'x'num_blocks' blocks, select the top half of the blocks with the largest 
    number of foreground pixels, and return the average spectral vector computed from 
    the pixels in each of these blocks.

    Parameters
    ----------
    mask : 2D np.ndarray of bool
        Cleaned binary mask (True = bacterium region, False = background).
    ms_data : 3D np.ndarray of shape (H, W, num_bands)
        Must match mask shape on the first two dimensions.
    num_blocks : int
        The number of blocks to divide the image into.

    Returns
    -------
    avg_spectra : np.ndarray of shape (k, num_bands)
        The average spectral vector for each selected block, where k is the number of top blocks.
        Returns an empty array if no block contains foreground pixels.
    """
    H, W = mask.shape

    # Determine block boundaries for a num_blocks x num_blocks grid.
    row_bounds = np.linspace(0, H, num_blocks + 1, dtype=int)
    col_bounds = np.linspace(0, W, num_blocks + 1, dtype=int)
    
    # For each block, compute the number of foreground pixels and store the block boundaries.
    blocks_info = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            r_start = row_bounds[i]
            r_end = row_bounds[i + 1]
            c_start = col_bounds[j]
            c_end = col_bounds[j + 1]
            block_mask = mask[r_start:r_end, c_start:c_end]
            count = np.sum(block_mask)
            blocks_info.append((count, r_start, r_end, c_start, c_end))
    
    # Sort blocks by the number of True pixels (descending order)
    sorted_blocks = sorted(blocks_info, key=lambda x: x[0], reverse=True)
    
    # Select the top half of the blocks
    num_top_blocks = round(num_blocks * num_blocks / 2 + 0.5)
    selected_blocks = sorted_blocks[:num_top_blocks]
    
    # For each selected block, compute the average spectral vector from the pixels within the block.
    avg_spectra_list = []
    for count, r_start, r_end, c_start, c_end in selected_blocks:
        # Ensure the block has foreground pixels.
        if count == 0:
            return None
        
        block_mask = mask[r_start:r_end, c_start:c_end]
        block_ms_data = ms_data[r_start:r_end, c_start:c_end, :]
        # Extract spectral vectors only where the block mask is True.
        pixels = block_ms_data[block_mask]
        # Compute the average spectrum for this block.
        block_avg = np.mean(pixels, axis=0, keepdims=True)  # shape: (1, num_bands)
        avg_spectra_list.append(block_avg)
    
    avg_spectra = np.vstack(avg_spectra_list)  # shape: (num_selected_blocks, num_bands)
    
    # Ensure the result is a 2D array.
    avg_spectra = np.array(avg_spectra).squeeze()
    if avg_spectra.ndim == 1:
        avg_spectra = avg_spectra.reshape(1, -1)
    
    # If the data has 260 dimensions, remove the first and last bands.
    if avg_spectra.shape[1] == 260:
        avg_spectra = avg_spectra[:, 1:-1]
    assert avg_spectra.shape[1] == NUM_BANDS, "Number of bands error!"
    
    return avg_spectra

 
def balanced_subsample(df, label_col='coarse_class_id', class_labels=None, max_class_size=300, random_state=42):
    """
    Downsample any class whose size exceeds `max_class_size` to `max_class_size`.
    Leaves smaller classes as is.

    Parameters
    ----------
    df : pd.DataFrame
        The original metadata DataFrame.
    label_col : str
        The name of the column containing class IDs (e.g., 'coarse_class_id').
    max_class_size : int
        The maximum allowed number of samples per class.
        Classes larger than this are downsampled to this size.
    random_state : int
        Random seed for reproducible sampling.
    class_labels : list or None
        If provided, only rows with values in this list (in the column specified by
        `label_col`) will be kept.

    Returns
    -------
    balanced_df : pd.DataFrame
        A DataFrame where no class exceeds `max_class_size` samples,
        shuffled and reset index.
    """

    # 0) Filter the DataFrame if class_labels is provided.
    if class_labels is not None:
        df = df[df[label_col].isin(class_labels)]

    # 1) Class distribution before balancing
    class_sizes = df.groupby(label_col).size()
    print("Class distribution before balancing:")
    print(class_sizes)
    
    df_list = []
    for cls_id, group_size in class_sizes.items():
        sub_df = df[df[label_col] == cls_id]
        # 2) Only downsample if this class exceeds max_class_size
        if group_size > max_class_size:
            sub_df = sub_df.sample(n=max_class_size, random_state=random_state)
        df_list.append(sub_df)
    
    # 3) Combine all classes back together and shuffle
    balanced_df = pd.concat(df_list, axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 4) Class distribution after balancing
    print("\nClass distribution after balancing:")
    print(balanced_df.groupby(label_col).size())

    # 5) Update coarse_class_id to start from 0 based on coarse_label
    unique_coarse = sorted(balanced_df["coarse_label"].unique())
    coarse_mapping = {label: i for i, label in enumerate(unique_coarse)}
    balanced_df["coarse_class_id"] = balanced_df["coarse_label"].map(coarse_mapping)

    # 6) Update fine_class_id to start from 0 based on fine_label
    unique_fine = sorted(balanced_df["fine_label"].unique())
    fine_mapping = {label: i for i, label in enumerate(unique_fine)}
    balanced_df["fine_class_id"] = balanced_df["fine_label"].map(fine_mapping)

    return balanced_df


def create_balanced_dataset_df(metadata_file, all_file, train_file, test_file, test_size=0.2, random_state=42, label_col='coarse_label', class_labels=None, max_class_size=300):
    # 1) Read the metadata
    df = pd.read_excel(metadata_file)
    
    # 2) Subsample the largest class to match the smallest
    balanced_df = balanced_subsample(df, label_col=label_col, class_labels=class_labels, max_class_size=max_class_size, random_state=42)
    
    # 3) Split into train & test
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=test_size,
        stratify=balanced_df[label_col],
        random_state=random_state
    )
    
    print("\nTrain set size:", len(train_df))
    print("Test set size:", len(test_df))
    
    print("\nTrain distribution:")
    print(train_df.groupby(label_col).size())
    
    print("\nTest distribution:")
    print(test_df.groupby(label_col).size())

    # save the all, train and test files
    balanced_df.to_excel(all_file, index=False)
    train_df.to_excel(train_file, index=False)
    test_df.to_excel(test_file, index=False)

    return balanced_df, train_df, test_df


# --------------------------------------------------
# 1) Define a helper function to process a single row
# --------------------------------------------------
def _process_single_image(row, data_root, background_subtraction, num_samples):
    """
    row: a Series or namedtuple from df.itertuples() / df.iterrows()
    data_root: base directory for the HDR files
    background_subtraction: whether to perform background subtraction
    num_samples: number of pixels to sample
    
    Returns a dictionary of fields consistent with the HF features schema
    """
    folder_path = row["folder_path_new"]
    hdr_filename = row["hdr_filename_new"]
    background_hdr_filename = row["background_hdr_filename_new"]
    coarse_label = row["coarse_label"]
    coarse_id = row["coarse_class_id"]
    fine_label = row["fine_label"]
    fine_id = row["fine_class_id"]

    full_path = os.path.join(data_root, folder_path, hdr_filename)
    background_full_path = os.path.join(data_root, folder_path, background_hdr_filename)
    
    # 1) Load the multispectral data (User-defined function)
    ms_data, header = load_ms_image(full_path)
    if ms_data is None:
        print(f"Skipping {full_path} with no valid data: load data failed")
        return None
    
    if background_subtraction:
        background_ms_data, background_header = load_ms_image(background_full_path)
        assert header['wavelength'] == background_header['wavelength'], f"Headers do not match for background subtraction:{row['folder_path']}:{row['hdr_filename']} vs. {row['background_hdr_filename']}"
        ms_data = ms_data - background_ms_data
        ms_data = ms_data - np.min(ms_data)  # ensure all values are non-negative
    
    # Compress the dynamic range
    ms_data = np.log10(ms_data + 1)

    # 2) Convert to grayscale (normalized to [0, 1])
    gray_image = convert_ms_to_gray(ms_data)
    
    # 3) Threshold, erosion, remove small regions, etc. 
    mask = binary_thresholding(gray_image)
    
    # 4) Uniformly sample from the mask 
    coords, spectra = sample_mask_pixels(mask, ms_data, num_samples, random_state=42)

    if spectra is None:
        print(f"Skipping {full_path} with no valid pixels")
        return None
    
    if spectra.shape[0] != num_samples:
        print(f"Skipping {full_path} with no valid pixels: num of pixels = {spectra.shape[0]}")
        return None

    return {
        "folder_path": folder_path,
        "hdr_filename": hdr_filename,
        "coarse_label": coarse_label,
        "coarse_class_id": coarse_id,
        "fine_label": fine_label,
        "fine_class_id": fine_id,
        "spectra": spectra  # shape (num_samples, NUM_BANDS)
    }


def _process_single_image_background_division(row, data_root, background_subtraction, num_blocks):
    """
    row: a Series or namedtuple from df.itertuples() / df.iterrows()
    data_root: base directory for the HDR files
    background_subtraction: whether to perform background subtraction
    num_samples: number of pixels to sample
    
    Returns a dictionary of fields consistent with the HF features schema
    """
    folder_path = row["folder_path_new"]
    hdr_filename = row["hdr_filename_new"]
    background_hdr_filename = row["background_hdr_filename_new"]
    coarse_label = row["coarse_label"]
    coarse_id = row["coarse_class_id"]
    fine_label = row["fine_label"]
    fine_id = row["fine_class_id"]

    full_path = os.path.join(data_root, folder_path, hdr_filename)
    background_full_path = os.path.join(data_root, folder_path, background_hdr_filename)
    
    # 1) Load the multispectral data (User-defined function)
    ms_data, header = load_ms_image(full_path)
    if ms_data is None:
        print(f"Skipping {full_path} with no valid data: load data failed")
        return None
    
    if background_subtraction:
        background_ms_data, background_header = load_ms_image(background_full_path)
        assert header['wavelength'] == background_header['wavelength'], f"Headers do not match for background subtraction:{row['folder_path']}:{row['hdr_filename']} vs. {row['background_hdr_filename']}"

        # Clip the background: ensure that every pixel value is at least 1.
        background_ms_data_clipped = np.maximum(background_ms_data, 1)
        # Perform element-wise division.
        ms_data = ms_data / background_ms_data_clipped
        ms_data = np.minimum(ms_data, 1)
    
    # Compress the dynamic range
    # ms_data = np.log10(ms_data + 1)

    # 2) Convert to grayscale (normalized to [0, 1])
    gray_image = convert_ms_to_gray(ms_data)
    
    # 3) Threshold, erosion, remove small regions, etc. 
    mask = binary_thresholding(gray_image)
    
    # 4) Sample from the mask using block averaging
    spectra = sample_mask_pixels_block_avg(mask, ms_data, num_blocks)

    if spectra is None:
        print(f"Skipping {full_path} with no valid pixels")
        return None
    
    return {
        "folder_path": folder_path,
        "hdr_filename": hdr_filename,
        "coarse_label": coarse_label,
        "coarse_class_id": coarse_id,
        "fine_label": fine_label,
        "fine_class_id": fine_id,
        "spectra": spectra  # shape (num_samples, NUM_BANDS)
    }


# --------------------------------------------------
# 2) Main function that parallelizes over all rows
# --------------------------------------------------
def preprocess_and_build_bacteria_dataset(df, data_root, background_subtraction=False, num_blocks=5, n_jobs=8):
    """
    Given a dataframe 'df' (train or test), load each multispectral image,
    perform preprocessing in parallel, and gather results into a Hugging Face Dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: coarse_label, coarse_class_id, fine_label, fine_class_id, 
        hdr_filename_new, folder_path_new, etc.
    data_root : str
        Root directory where the HDR files reside.
    background_subtraction : bool
        Whether to perform background subtraction.
    num_blocks : int
        Number of blocks to use for block averaging, total blocks will be num_blocks * num_blocks.
    n_jobs : int
        Number of CPU cores to use for parallel processing (-1 uses all cores).
    
    Returns
    -------
    hf_dataset : datasets.Dataset
        Hugging Face Dataset containing the preprocessed data.
    """
    # 1) Define a Hugging Face schema, with shape = (num_samples, NUM_BANDS)
    num_samples = round(num_blocks * num_blocks /2 + 0.5)
    features = Features({
        "folder_path": Value("string"),
        "hdr_filename": Value("string"),
        "coarse_label": Value("string"),
        "coarse_class_id": Value("int32"),
        "fine_label": Value("string"),
        "fine_class_id": Value("int32"),
        "spectra": Array2D(shape=(num_samples, NUM_BANDS), dtype="float32")
    })
    
    # 2) Convert list of dicts to a single dict of lists (data_dict)
    #    Each key in the features schema will map to a list of values
    data_dict = {key: [] for key in features.keys()}

    # 3) Parallelize row processing
    #    - tqdm for a nice progress bar
    #    - joblib.Parallel with delayed calls
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_image_background_division)(row, data_root, background_subtraction, num_blocks)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images")
    )
    
    # Explicitly filter out None results.
    valid_results = [item for item in results if item is not None]
    print(f"Number of valid images: {len(valid_results)} out of {len(df)}")

    # Build data_dict only from valid results.
    data_dict = {key: [] for key in features.keys()}
    for item in valid_results:
        for key in features.keys():
            data_dict[key].append(item[key])
    
    # 4) Create the Hugging Face dataset
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    return hf_dataset


def create_hf_bacteria_dataset(df, data_root, background_subtraction=False, num_blocks=5, out_dir="hf_dataset"):
    """
    Save dataset to disk in the Hugging Face Arrow format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: coarse_label, coarse_class_id, fine_label, fine_class_id, 
        hdr_filename_new, folder_path_new, etc.
    data_root : str
        Root directory for HDR files.
    num_blocks : int
        Number of blocks to use for block averaging, total blocks will be num_blocks * num_blocks.
    background_subtraction : bool
        Whether to perform background subtraction.
    out_dir : str
        Directory to save the final Arrow dataset.
    """
    # Build the dataset
    dataset = preprocess_and_build_bacteria_dataset(df, data_root, background_subtraction, num_blocks)
    
    # Combine them into a DatasetDict
    dataset_dict = DatasetDict({"dataset": dataset})
    
    # Save to disk in Hugging Face Arrow format
    dataset_dict.save_to_disk(out_dir)
    print(f"Saved Hugging Face dataset to {out_dir}")


def compute_spectra_mean_std(train_dataset):
    """
    Compute the per-band mean and standard deviation for the 'spectra'
    field in a Hugging Face dataset.
    
    Each sample in the dataset is expected to have a 'spectra' field of shape (N, D),
    where N is the number of pixels (e.g., 500) and D is the number of bands (e.g., 258).
    
    The function concatenates all spectra from all samples along the pixel dimension,
    and then computes the mean and std for each band.
    
    Parameters
    ----------
    train_dataset : datasets.Dataset
        The training dataset from Hugging Face containing a "spectra" field.
    
    Returns
    -------
    mean : np.ndarray of shape (D,)
        Per-band mean computed across all pixels in all samples.
    std : np.ndarray of shape (D,)
        Per-band standard deviation computed across all pixels in all samples.
    """
    all_spectra = []  # to collect each sample's spectra array
    for sample in train_dataset:
        # Convert the 'spectra' field (huggingface Array2D) to a NumPy array.
        spectra_arr = np.array(sample["spectra"])  # shape: (N, D)
        all_spectra.append(spectra_arr)
    
    # Concatenate along the pixel axis: result shape is (num_samples * N, D)
    all_spectra = np.concatenate(all_spectra, axis=0)
    
    # Compute per-band statistics (i.e., along axis=0 which aggregates all pixels)
    mean = np.mean(all_spectra, axis=0)
    std = np.std(all_spectra, axis=0)
    
    return mean, std


def standardize_features(X, mean, std):
    """
    Standardize X by subtracting mean and dividing by std.
    """
    return (X - mean) / (std + 1e-8)


def process_sample(row, data_root, background_subtraction=False, num_samples=100, random_state=42):
    """
    Process one image based on the dataframe row.
    
    Parameters
    ----------
    row : pd.Series
        A row from the dataframe (e.g., from train_df) with keys:
            "folder_path_new", "hdr_filename_new", etc.
    data_root : str
        Root directory where image files are stored.
    background_subtraction : bool
        Whether to subtract the background image from the sample image.
    num_samples : int
        Number of pixels to sample from the mask.
    random_state : int
        Random seed for reproducible sampling.
    
    Returns
    -------
    sample_dict : dict
        Dictionary containing:
            - "gray": Grayscale image (2D array).
            - "masked_gray": Grayscale image with only mask pixels.
            - "coords": (n, 2) array of pixel coordinates sampled.
            - "spectra": 2D array of sampled pixel spectra (shape: (num_samples, NUM_BANDS)).
    """
    folder_path = row["folder_path_new"]
    hdr_filename = row["hdr_filename_new"]
    background_hdr_filename = row["background_hdr_filename_new"]

    full_path = os.path.join(data_root, folder_path, hdr_filename)
    background_full_path = os.path.join(data_root, folder_path, background_hdr_filename)
    
    # Load the multispectral image and header.
    ms_data, header = load_ms_image(full_path)
    if ms_data is None:
        print(f"Skipping {full_path} with no valid data: load data failed")
        return None
    
    if background_subtraction:
        background_ms_data, background_header = load_ms_image(background_full_path)
        assert header['wavelength'] == background_header['wavelength'], f"Headers do not match for background subtraction:{row['folder_path']}:{row['hdr_filename']} vs. {row['background_hdr_filename']}"
        ms_data = ms_data - background_ms_data
        ms_data = ms_data - np.min(ms_data)  # ensure all values are non-negative
    
    # Compress the dynamic range
    ms_data = np.log10(ms_data + 1)

    # Convert to grayscale.
    gray_image = convert_ms_to_gray(ms_data)
    
    # Apply binary thresholding to get a mask.
    mask = binary_thresholding(gray_image)
    
    # Uniformly sample pixels from the mask.
    coords, spectra = sample_mask_pixels(mask, ms_data, num_samples, random_state=random_state)
    
    # Create a masked grayscale image: pixels outside the mask are zero.
    masked_gray = np.zeros_like(gray_image)
    masked_gray[mask] = gray_image[mask]
    
    sample_dict = {
        "filename": os.path.join(folder_path, hdr_filename),
        "gray": gray_image,
        "masked_gray": masked_gray,
        "coords": coords,      # Expected shape: (num_samples, 2)
        "spectra": spectra     # Expected shape: (num_samples, NUM_BANDS)
    }
    return sample_dict


def process_sample_background_division(row, data_root, background_subtraction=False):
    """
    Process one image based on the dataframe row.
    
    Parameters
    ----------
    row : pd.Series
        A row from the dataframe (e.g., from train_df) with keys:
            "folder_path_new", "hdr_filename_new", etc.
    data_root : str
        Root directory where image files are stored.
    background_subtraction : bool
        Whether to subtract the background image from the sample image.
    
    Returns
    -------
    sample_dict : dict
        Dictionary containing:
            - "gray": Grayscale image (2D array).
            - "masked_gray": Grayscale image with only mask pixels.
            - "coords": (n, 2) array of pixel coordinates sampled.
            - "spectra": 2D array of sampled pixel spectra (shape: (num_samples, NUM_BANDS)).
    """
    folder_path = row["folder_path_new"]
    hdr_filename = row["hdr_filename_new"]
    background_hdr_filename = row["background_hdr_filename_new"]

    full_path = os.path.join(data_root, folder_path, hdr_filename)
    background_full_path = os.path.join(data_root, folder_path, background_hdr_filename)
    
    # Load the multispectral image and header.
    ms_data, header = load_ms_image(full_path)
    if ms_data is None:
        print(f"Skipping {full_path} with no valid data: load data failed")
        return None
    
    if background_subtraction:
        background_ms_data, background_header = load_ms_image(background_full_path)
        assert header['wavelength'] == background_header['wavelength'], f"Headers do not match for background subtraction:{row['folder_path']}:{row['hdr_filename']} vs. {row['background_hdr_filename']}"

        # Clip the background: ensure that every pixel value is at least 1.
        background_ms_data_clipped = np.maximum(background_ms_data, 1)
        # Perform element-wise division.
        ms_data = ms_data / background_ms_data_clipped
        ms_data = np.minimum(ms_data, 1)

        # ms_data = np.nan_to_num(ms_data, nan=0.0)
    
    # Compress the dynamic range
    # ms_data = np.log10(ms_data + 1)

    # Convert to grayscale.
    gray_image = convert_ms_to_gray(ms_data)
    
    # Apply binary thresholding to get a mask.
    mask = binary_thresholding(gray_image)
    
    # 4) Sample from the mask using block averaging
    spectra = sample_mask_pixels_block_avg(mask, ms_data, num_blocks=5)
    
    # Create a masked grayscale image: pixels outside the mask are zero.
    masked_gray = np.zeros_like(gray_image)
    masked_gray[mask] = gray_image[mask]
    
    sample_dict = {
        "filename": os.path.join(folder_path, hdr_filename),
        "gray": gray_image,
        "masked_gray": masked_gray,
        "spectra": spectra     # Expected shape: (num_samples, NUM_BANDS)
    }
    return sample_dict


def visualize_samples(processed_samples, fig_name=""):
    """
    Visualize processed image samples in a 4xN grid.
    
    Rows represent:
      Row 1: Grayscale image.
      Row 2: Masked grayscale image.
      Row 3: Masked grayscale image with sampled pixels (red dots).
      Row 4: The sampled spectra displayed as an image.
    
    Parameters
    ----------
    processed_samples : list of dict
        List of dictionaries from process_sample.
        Length determines the number of columns.
    """
    num_samples = len(processed_samples)
    
    # Create a figure with 4 rows and num_samples columns.
    fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 16))
    
    # If there's only one sample, axes will be 1D; make sure it's 2D.
    if num_samples == 1:
        axes = axes[:, np.newaxis]
    
    for j, sample in enumerate(processed_samples):
        # Row 1: Grayscale image.
        ax = axes[0, j]
        ax.imshow(sample["gray"], cmap="gray")
        ax.set_title(f"{sample['filename'].split('.')[0]}")
        ax.axis("off")
        
        # Row 2: Masked grayscale image.
        ax = axes[1, j]
        ax.imshow(sample["masked_gray"], cmap="gray")
        ax.set_title("Masked Gray")
        ax.axis("off")
        
        # # Row 2: Overlay sampled pixels on masked grayscale.
        # ax = axes[1, j]
        # ax.imshow(sample["masked_gray"], cmap="gray")
        # # Ensure coords is a NumPy array.
        # coords = np.array(sample["coords"])
        # if coords.size > 0:
        #     ax.scatter(coords[:, 1], coords[:, 0], c="red", s=5)
        # ax.set_title("Sampled Pixels")
        # ax.axis("off")
        
        # Row 3: Display the spectra as an image.
        ax = axes[2, j]
        # Here, we assume sample["spectra"] is a 2D array of shape (num_samples, NUM_BANDS).
        # Use aspect="auto" so all bands are visible.
        im = ax.imshow(sample["spectra"], aspect="auto", cmap="viridis")
        ax.set_title("Spectra")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 4: Display the spectra as a line plot.
        ax = axes[3, j]
        # Here, we assume sample["spectra"] is a 2D array of shape (num_samples, NUM_BANDS).
        for i in range(NUM_BANDS):
            ax.plot(np.mean(sample["spectra"], axis=0))
        ax.set_title("Avg. Spectra")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Intensity")
    
    plt.tight_layout()

    if fig_name == "":
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches="tight")
        plt.close()


def get_label_mapping_from_dataset(dataset, classification_level = "coarse"):
    """
    Extract a mapping from coarse_class_id (int) to coarse_label (str)
    from a Hugging Face dataset.
    
    Assumes each sample has fields 'coarse_class_id' and 'coarse_label'.
    If multiple samples share the same coarse_class_id, their coarse_label 
    is assumed to be the same.
    
    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset (e.g. train_dataset) containing the fields.
    classification_level : str, optional
        The classification level to use. Can be "coarse" or "fine".
        
    Returns
    -------
    mapping : dict
        Dictionary mapping coarse_class_id (int) to coarse_label (str).
    """
    mapping = {}
    for sample in dataset:
        if classification_level == "fine":
            cid = sample["fine_class_id"]
            label = sample["fine_label"]
        else:
            cid = sample["coarse_class_id"]
            label = sample["coarse_label"]

        # If already exists, you could check for consistency if desired.
        mapping[cid] = label

    return mapping


def display_classification_results(y_test, y_pred, label_mapping=None):
    """
    Compute and display evaluation metrics: precision, recall, F1-score,
    and the confusion matrix. Also plot a heatmap of the confusion matrix
    and a bar chart of per-class metrics with values annotated on top of bars.
    
    Parameters
    ----------
    y_test : np.ndarray
        True labels (int, 0-based).
    y_pred : np.ndarray
        Predicted labels (int, 0-based).
    label_mapping : dict or None
        A dictionary mapping integer class IDs to coarse class labels (str).
        For example: {0: "Bacteria Type A", 1: "Bacteria Type B", ...}.
        If provided, class names will be used in the printed report and plots.
    """
    # Compute the classification report as a dictionary.
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print(report)
    
    # Print overall weighted metrics.
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # print(f"Weighted Precision: {precision:.3f}")
    # print(f"Weighted Recall: {recall:.3f}")
    # print(f"Weighted F1: {f1:.3f}")
    
    # Compute confusion matrix.
    if label_mapping is not None:
        # Get sorted class IDs and corresponding labels.
        unique_labels = sorted(label_mapping.keys())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        tick_labels = [label_mapping[i] for i in unique_labels]
    else:
        cm = confusion_matrix(y_test, y_pred)
        tick_labels = None
    
    # Plot confusion matrix as a heatmap.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    
    # Prepare per-class metrics for plotting (ignore aggregate entries).
    class_keys = [k for k in report.keys() if k.isdigit()]
    if label_mapping is not None:
        classes_labels = [label_mapping[int(k)] for k in class_keys]
    else:
        classes_labels = class_keys
    precisions = [report[k]["precision"] for k in class_keys]
    recalls = [report[k]["recall"] for k in class_keys]
    f1s = [report[k]["f1-score"] for k in class_keys]
    
    x = np.arange(len(class_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width, precisions, width, label='Precision')
    bars2 = ax.bar(x, recalls, width, label='Recall')
    bars3 = ax.bar(x + width, f1s, width, label='F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_labels, rotation=0, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Evaluation Metrics")
    ax.legend()
    
    # Annotate bars with their values.
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., 
                height,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.show()


def extract_features(dataset, feature_method="average", classification_level="coarse"):
    """
    Extract features from a Hugging Face dataset containing multispectral data.
    
    Parameters
    ----------
    dataset : datasets.Dataset
        A Hugging Face dataset (e.g., train or test split) with at least the following fields:
            - "spectra": a 2D array (list-of-lists) with shape (n_pixels, NUM_BANDS).
            - "coarse_class_id": an integer (0-based) label used for classification.
            - "coarse_label": the image label as a string.
    feature_method : str, optional
        Feature extraction method:
            "average":   Average the pixel spectra to yield one feature vector per image.
                         Returns X with shape (n_images, NUM_BANDS) and y with shape (n_images,).
            "pixelwise": Treat every pixel's spectrum as an independent sample.
                         Returns X with shape (total_pixels, NUM_BANDS), y with shape (total_pixels,),
                         and an image_ids array of shape (total_pixels,) that indicates the image
                         from which each pixel originates.
    classification_level : str, optional
        The level of classification used in the dataset. Can be "coarse" or "fine".
    
    Returns
    -------
    X : np.ndarray
        Feature matrix. If method=="average", shape = (n_images, NUM_BANDS); 
        if method=="pixelwise", shape = (total_pixels, NUM_BANDS).
    y : np.ndarray
        Label vector. If method=="average", length = n_images;
        if method=="pixelwise", length = total_pixels.
    image_ids : np.ndarray or None
        If method=="pixelwise", an array of integers of length total_pixels, where each entry
        indicates the image index (starting from 0) that the pixel belongs to.
        If method=="average", returns None.
    """
    if feature_method == "average":
        # Each image yields one feature vector (the average spectrum)
        avg_features = []  # list of (NUM_BANDS,) vectors
        labels = []        # list of int labels (one per image)
        
        for sample in dataset:
            # Convert the stored 'spectra' (list-of-lists) into a NumPy array.
            spectra = np.array(sample["spectra"])  # shape: (n_pixels, NUM_BANDS)
            # Average the spectra over all pixels (axis 0)
            avg_spectrum = np.mean(spectra, axis=0)
            avg_features.append(avg_spectrum)

            if classification_level == "fine":
                # Use fine_class_id as the label (should already be int)
                labels.append(sample["fine_class_id"])
            else:
                # Use coarse_class_id as the label (should already be int)
                labels.append(sample["coarse_class_id"])
        
        X = np.vstack(avg_features)  # shape: (n_images, NUM_BANDS)
        y = np.array(labels)         # shape: (n_images,)
        image_ids = None

    elif feature_method == "pixelwise":
        # Every pixel's spectrum is an independent sample.
        pixel_features = []  # list of arrays, each of shape (n_pixels, NUM_BANDS)
        pixel_labels = []    # list of 1D arrays, one label per pixel
        pixel_image_ids = [] # list of 1D arrays, one image id per pixel
        
        image_counter = 0
        for sample in dataset:
            spectra = np.array(sample["spectra"])  # shape: (n_pixels, NUM_BANDS)
            pixel_features.append(spectra)
            n_pixels = spectra.shape[0]
            # Replicate the image label (coarse_class_id) for every pixel
            if classification_level == "fine":
                labels_per_sample = np.full(n_pixels, sample["fine_class_id"], dtype=int)
            else:
                labels_per_sample = np.full(n_pixels, sample["coarse_class_id"], dtype=int)
            pixel_labels.append(labels_per_sample)
            # Create an image id array for this sample
            image_ids_per_sample = np.full(n_pixels, image_counter, dtype=int)
            pixel_image_ids.append(image_ids_per_sample)
            image_counter += 1
        
        # Concatenate all pixels from all images along axis=0.
        X = np.concatenate(pixel_features, axis=0)  # shape: (total_pixels, NUM_BANDS)
        y = np.concatenate(pixel_labels, axis=0)      # shape: (total_pixels,)
        image_ids = np.concatenate(pixel_image_ids, axis=0)  # shape: (total_pixels,)
    
    else:
        raise ValueError("Method must be either 'average' or 'pixelwise'.")
    
    return X, y, image_ids


def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM classifier on the training data and evaluate on the test data.
    Also, display evaluation metrics and plots.
    
    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, D)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, D)
    y_test  : np.ndarray of shape (n_test,)
    
    Returns
    -------
    clf : trained SVC classifier
    y_pred : np.ndarray of predictions on X_test
    """
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    display_classification_results(y_test, y_pred)
    
    return clf, y_pred


def svm_classification_pipeline(train_dataset, test_dataset, train_mean, train_std, feature_method='average', classification_level='coarse'):
    """
    Complete SVM classification pipeline:
      1. Extract features from both train and test datasets.
      2. Normalize the features.
      3. Train an SVM classifier.
      4. Evaluate the classifier.
         For pixelwise features, fuse pixel predictions into image-level predictions.
    
    Parameters
    ----------
    train_dataset : datasets.Dataset
        Hugging Face dataset for training.
    test_dataset : datasets.Dataset
        Hugging Face dataset for testing.
    train_mean : np.ndarray
        Per-band mean from the training data (shape (D,)).
    train_std : np.ndarray
        Per-band standard deviation from the training data (shape (D,)).
    feature_method : str, optional
        Feature extraction method:
            "average":   Average the pixel spectra to yield one feature vector per image.
                         Returns X with shape (n_images, NUM_BANDS) and y with shape (n_images,).
            "pixelwise": Treat every pixel's spectrum as an independent sample.
                         Returns X with shape (total_pixels, NUM_BANDS), y with shape (total_pixels,),
                         and image_ids with shape (total_pixels,) indicating the originating image.
    classification_level : str, optional
        The level of classification used in the dataset. Can be "coarse" or "fine".

    Returns
    -------
    clf : sklearn.svm.SVC
        The trained SVM classifier.
    y_pred : np.ndarray
        The predicted labels at the pixel level (if method=="pixelwise") or image level (if "average").
    label_mapping : dict
        A mapping from numeric labels to the original class names.
    """
    
    # Extract label mapping (assumed available)
    label_mapping = get_label_mapping_from_dataset(train_dataset, classification_level)
    
    # Extract features. For pixelwise method, expect an extra image_ids array.
    if feature_method == "pixelwise":
        X_train, y_train, image_ids_train = extract_features(train_dataset, feature_method=feature_method, classification_level=classification_level)
        X_test, y_test, image_ids_test = extract_features(test_dataset, feature_method=feature_method, classification_level=classification_level)
    elif feature_method == "average":
        X_train, y_train, _ = extract_features(train_dataset, feature_method=feature_method, classification_level=classification_level)
        X_test, y_test, _ = extract_features(test_dataset, feature_method=feature_method, classification_level=classification_level)
    else:
        raise ValueError("Method must be either 'average' or 'pixelwise'.")
    
    # Normalize the features.
    X_train = standardize_features(X_train, train_mean, train_std)
    X_test = standardize_features(X_test, train_mean, train_std)    
    
    print("Training SVM classifier...\n")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on test set (pixel-level predictions if method=="pixelwise")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if feature_method == "pixelwise":
        print(f"Pixelwise Classification")
    else:
        print(f"Average-feature Classification")
    print(f"Test Accuracy: {acc:.3f}")
    display_classification_results(y_test, y_pred, label_mapping=label_mapping)

    if feature_method == "pixelwise":
        # Fusion: for each image, use majority voting over its pixels.
        unique_ids = np.unique(image_ids_test)
        fused_y_pred = []
        fused_y_true = []
        for uid in unique_ids:
            indices = np.where(image_ids_test == uid)[0]
            pixel_preds = y_pred[indices]
            # Majority vote: choose the most common prediction.
            fused_label = np.bincount(pixel_preds).argmax()
            fused_y_pred.append(fused_label)
            # Assume all pixels in an image share the same ground truth label.
            fused_y_true.append(y_test[indices[0]])
        fused_y_pred = np.array(fused_y_pred)
        fused_y_true = np.array(fused_y_true)
        
        acc = accuracy_score(fused_y_true, fused_y_pred)
        print(f"Image-level Test Accuracy: {acc:.3f}")
        display_classification_results(fused_y_true, fused_y_pred, label_mapping=label_mapping)
        return clf, fused_y_true, fused_y_pred
    else:
        return clf, y_test, y_pred


# Testing the functions
# Need to use ipython to show the figures
# In terminal run: ipython --pylab, then run your_script.py
# In VS Code, add the following to your launch.json: Run -> Run Curent File in Interactive Window
if __name__ == '__main__':

    metadata_file = "Reorganized_Bacteria_Data/bacteria_metadata_reorganized.xlsx"
    train_file    = "Reorganized_Bacteria_Data/balanced_train_data.xlsx"
    test_file     = "Reorganized_Bacteria_Data/balanced_test_data.xlsx"
    data_root     = "Reorganized_Bacteria_Data"

    # Create balanced train and test datalists
    # train_df, test_df = create_balanced_dataset_df(metadata_file, train_file, test_file, test_size=0.2, random_state=42)

    # Load the train and test dataframes from disk
    train_df = pd.read_excel(train_file)
    test_df  = pd.read_excel(test_file)

    # Build the Hugging Face dataset, just onece
    # create_hf_bacteria_dataset(train_df, test_df, data_root, num_samples=100, out_dir="bacteria_dataset_balanced")

    dataset = load_from_disk('bacteria_dataset_balanced')

    train_dataset = dataset['train']

    for i, sample in enumerate(train_dataset):
        folder_path = sample["folder_path"]
        hdr_filename = sample["hdr_filename"]        
        spectra = np.array(sample["spectra"])
        if spectra.size != 500 * 258:
            print(f"Sample {i} has incorrect size: {spectra.size}")

    mean, std = compute_spectra_mean_std(train_dataset)
    print("Per-band mean:", mean)
    print("Per-band std:", std)


    # Load the image and header
    row = train_df.iloc[8]
    folder_path = row["folder_path_new"]
    hdr_filename = row["hdr_filename_new"]
    full_path = os.path.join(data_root, folder_path, hdr_filename)
    ms_data, header = load_ms_image(full_path)
    print(header)

    # Convert the multispectral image to grayscale
    gray_image = convert_ms_to_gray(ms_data)

    # Apply binary thresholding
    mask = binary_thresholding(gray_image)

    # Sample 500 pixels from the mask
    num_samples = 500
    coords, spectra = sample_mask_pixels(mask, ms_data, num_samples, random_state=42)
    print("Sampled pixels:", spectra.shape)
    assert len(spectra) == num_samples
    

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    ax[0].imshow(gray_image, cmap='gray')
    ax[0].set_title("Grayscale Image")
    ax[0].axis('off')
    
    masked_gray = np.zeros_like(gray_image)
    masked_gray[mask] = gray_image[mask]
    ax[1].imshow(masked_gray, cmap='gray')
    ax[1].set_title("Masked Grayscale Image")
    ax[1].axis('off')
    
    ax[2].imshow(masked_gray, cmap='gray')
    ax[2].set_title("Sampled Pixels")
    if coords.size > 0:
        ax[2].scatter(coords[:, 1], coords[:, 0], c='red', s=1)
    ax[2].axis('off')
    
    print("Done!")