# -*- coding: utf-8 -*-
"""
This script processes binary segmentation masks to generate bounding boxes for objects of interest.
It reads a folder of segmentation masks, identifies clusters of specified pixels, and calculates
the bounding boxes for these clusters. The script then draws these bounding boxes on the
corresponding original images and saves the annotated images. It also saves the coordinates
of the bounding boxes to a JSON file.

The script uses the DBSCAN clustering algorithm to group pixels and is optimized for
multithreaded processing to handle large numbers of images efficiently.

The saved JSON file contains the pixel coordinates for each bounding box. The coordinates are
for the top-left (x_min, y_min) and bottom-right (x_max, y_max) corners of the rectangle.
The format for each box is [[x_min, y_min], [x_max, y_max]].

Dependencies:
    - opencv-python
    - numpy
    - scikit-learn

Usage:
    python pixel2bbox.py --mask_folder /path/to/masks \
                         --image_folder /path/to/images \
                         --output_folder /path/to/output \
                         --label_name "ObjectOfInterest" \
                         --pixel_intensity 1 \
                         --eps 160 \
                         --min_samples 10 \
                         --area_threshold 100

"""
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import List, Dict, Tuple, Any

def get_pixel_clusters(
    image: np.ndarray,
    pixel_intensity: int,
    eps: int = 5,
    min_samples: int = 2
) -> Dict[int, List[np.ndarray]]:
    """
    Clusters pixels of a specific intensity in an image using DBSCAN.

    Args:
        image (np.ndarray): The input grayscale image.
        pixel_intensity (int): The pixel value to be clustered (e.g., 255 for white).
        eps (int): The maximum distance between two samples for one to be considered
                   as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.

    Returns:
        Dict[int, List[np.ndarray]]: A dictionary where keys are cluster labels and
                                     values are lists of pixel coordinates (x, y)
                                     belonging to that cluster.
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get coordinates of all pixels with the specified intensity
    y_coords, x_coords = np.where(image == pixel_intensity)
    pixel_coords = np.column_stack((x_coords, y_coords))

    # Return empty if no target pixels are found
    if len(pixel_coords) == 0:
        return {}

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixel_coords)
    labels = db.labels_

    # Organize pixels by cluster label
    clusters = {}
    for label, coord in zip(labels, pixel_coords):
        if label != -1:  # Ignore noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(coord)

    return clusters

def get_bounding_boxes(
    clusters: Dict[int, List[np.ndarray]],
    max_boxes: int = 3,
    area_threshold: int = 100
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Calculates bounding boxes for pixel clusters, filtered by area and count.

    Args:
        clusters (Dict[int, List[np.ndarray]]): A dictionary of pixel clusters.
        max_boxes (int): The maximum number of bounding boxes to return, sorted by area.
        area_threshold (int): The minimum area for a bounding box to be considered valid.

    Returns:
        List[Tuple[Tuple[int, int], Tuple[int, int]]]: A list of bounding boxes,
                                                       each defined by its top-left and
                                                       bottom-right coordinates.
    """
    bounding_boxes = []
    for cluster_pixels in clusters.values():
        if len(cluster_pixels) > 0:
            cluster_array = np.array(cluster_pixels)
            x_min, y_min = cluster_array.min(axis=0)
            x_max, y_max = cluster_array.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)

            # Add box to list if its area exceeds the threshold
            if area > area_threshold:
                bounding_boxes.append((((x_min, y_min), (x_max, y_max)), area))

    # Sort boxes by area in descending order and select the top `max_boxes`
    bounding_boxes.sort(key=lambda x: x[1], reverse=True)
    return [box[0] for box in bounding_boxes[:max_boxes]]

def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> np.ndarray:
    """
    Draws bounding boxes on an image.

    Args:
        image (np.ndarray): The image to draw on.
        bounding_boxes (List[Tuple[Tuple[int, int], Tuple[int, int]]]): A list of boxes to draw.

    Returns:
        np.ndarray: The image with bounding boxes drawn.
    """
    for top_left, bottom_right in bounding_boxes:
        cv2.rectangle(image, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2)
    return image

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_single_image(
    filename: str,
    mask_folder: str,
    image_folder: str,
    output_folder: str,
    label_name: str,
    pixel_intensity: int,
    eps: int,
    min_samples: int,
    max_boxes: int,
    area_threshold: int
) -> None:
    """
    Processes a single image: finds clusters, gets bounding boxes,
    draws them, and saves the output image and JSON data.
    """
    if not filename.endswith('.png'):
        print(f"The mask image should be saved as a png image, not {filename}.")
        return

    # Read the segmentation mask
    mask_path = os.path.join(mask_folder, filename)
    segmentation_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if segmentation_mask is None:
        print(f"Warning: Could not read mask file {mask_path}. Skipping.")
        return

    # Find pixel clusters
    clusters = get_pixel_clusters(segmentation_mask, pixel_intensity, eps, min_samples)

    if not clusters:
        print(f"Info: No pixel clusters found in {filename}. Skipping.")
        return

    # Get bounding boxes
    bounding_boxes = get_bounding_boxes(clusters, max_boxes, area_threshold)

    if not bounding_boxes:
        print(f"Info: No bounding boxes met the criteria for {filename}. Skipping.")
        return

    # Read the original image
    original_image_path = os.path.join(image_folder, filename)
    if not os.path.exists(original_image_path):
        print(f"Warning: Original image not found at {original_image_path}. Skipping.")
        return
    original_image = cv2.imread(original_image_path)

    # Draw bounding boxes on the original image
    annotated_image = draw_bounding_boxes(original_image.copy(), bounding_boxes)

    # Save the annotated image
    output_image_path = os.path.join(output_folder, 'annotated_images', filename)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Successfully saved annotated image to {output_image_path}")

    # Save bounding box coordinates to a JSON file
    json_data = {
        "image_filename": filename,
        label_name: bounding_boxes
    }

    json_filename = os.path.splitext(filename)[0] + '.json'
    json_output_path = os.path.join(output_folder, 'json_annotations', json_filename)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    print(f"Successfully saved bounding box coordinates to {json_output_path}")

def process_images_concurrently(
    mask_folder: str,
    image_folder: str,
    output_folder: str,
    label_name: str,
    pixel_intensity: int,
    eps: int,
    min_samples: int,
    max_boxes: int,
    max_workers: int,
    area_threshold: int
) -> None:
    """
    Processes all images in a folder using a thread pool.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    filenames = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                filename,
                mask_folder,
                image_folder,
                output_folder,
                label_name,
                pixel_intensity,
                eps,
                min_samples,
                max_boxes,
                area_threshold
            ) for filename in filenames
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing an image: {e}")

def main():
    """
    Main function to parse arguments and start the image processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes from segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mask_folder', type=str, required=True,
                        help="Path to the folder containing segmentation mask images.")
    parser.add_argument('--image_folder', type=str, required=True,
                        help="Path to the folder containing the original images.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Path to the folder where output will be saved.")
    parser.add_argument('--label_name', type=str, required=True,
                        help="Name for the object class to be used as a key in the output JSON.")
    parser.add_argument('--pixel_intensity', type=int, default=255,
                        help="The pixel intensity value of the mask to be detected.")
    parser.add_argument('--eps', type=int, default=50,
                        help="Epsilon value for DBSCAN clustering.")
    parser.add_argument('--min_samples', type=int, default=10,
                        help="Minimum number of samples for a DBSCAN cluster.")
    parser.add_argument('--max_boxes', type=int, default=10,
                        help="Maximum number of bounding boxes to save per image, sorted by area.")
    parser.add_argument('--area_threshold', type=int, default=100,
                        help="Minimum area of a bounding box to be considered valid.")
    parser.add_argument('--max_workers', type=int, default=2,
                        help="Maximum number of worker threads for parallel processing.")

    args = parser.parse_args()

    process_images_concurrently(
        mask_folder=args.mask_folder,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        label_name=args.label_name,
        pixel_intensity=args.pixel_intensity,
        eps=args.eps,
        min_samples=args.min_samples,
        max_boxes=args.max_boxes,
        max_workers=args.max_workers,
        area_threshold=args.area_threshold
    )

if __name__ == "__main__":
    main()