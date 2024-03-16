from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
import os


DATASET_ROOT = "/home/ubuntu/gamutRF-yolov8/SAM-fine-tune/dataset/fpv_yolo_manual_ds/"
TRAIN_DATASET_PATH = os.path.join(DATASET_ROOT, "train")
VAL_DATASET_PATH = os.path.join(DATASET_ROOT, "val")
IMAGES_DIR = "images"
LABELS_DIR = "labels"
MASKS_DIR = "masks"


def read_yolo_label(file_path: str) -> List[List[float]]:
    """Reads a YOLO label file and returns a list of bounding boxes.

    Args:
        file_path (str): File path of the YOLO label file.

    Returns:
        List[List[str]]: A list of lists, where each inner list represents a bounding box with
        its coordinates and label.

    Raises:
        FileNotFoundError: If the file specified by 'file_path' does not exist.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    bounding_boxes = []
    for line in lines:
        bounding_box = line.split()
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def get_split_bbox_coords(
    dataset_labels_path: str,
    dataset_images_path: str,
) -> Dict[str, np.ndarray]:
    """
    Reads a YOLO-formatted label file and returns the bounding box coordinates as a
    dictionary with keys being the label names (image filenames without extension).

    Args:
        dataset_labels_path (str): Path to the directory containing YOLO-formatted labels.
        image_width (int, optional): Width of the images in pixels. Defaults to IMG_WIDTH.
        image_height (int, optional): Height of the images in pixels. Defaults to IMG_HEIGHT.

    Returns:
        dict[str, np.ndarray]: Dictionary containing bounding box coordinates for each image
        as numpy arrays.

    Raises:
        FileNotFoundError: If the dataset_labels_path does not exist.
    """
    bbox_coords = {}

    for f in Path(dataset_labels_path).iterdir():
        k = f.stem

        image_height, image_width, _channels = cv2.imread(
            os.path.join(dataset_images_path, k + ".png")
        ).shape

        bounding_boxes = read_yolo_label(f)
        for bounding_box in bounding_boxes:
            xc = float(bounding_box[1])
            yc = float(bounding_box[2])
            w = float(bounding_box[3])
            h = float(bounding_box[4])
            x1 = xc - w / 2
            y1 = yc + h / 2
            x2 = xc + w / 2
            y2 = yc - h / 2
            if k not in bbox_coords.keys():
                bbox_coords[k] = np.array(
                    [
                        [
                            x1 * image_width,
                            y1 * image_height,
                            x2 * image_width,
                            y2 * image_height,
                        ]
                    ]
                )
            else:
                bbox_coords[k] = np.vstack(
                    (
                        bbox_coords[k],
                        np.array(
                            [
                                x1 * image_width,
                                y1 * image_height,
                                x2 * image_width,
                                y2 * image_height,
                            ]
                        ),
                    ),
                )
    return bbox_coords


def convert_bbox_to_masks(
    bbox_coords: Dict[str, np.ndarray],
    dataset_images_path: str,
) -> Dict[str, np.ndarray]:
    """
    Convert bounding box coordinates to mask representations.

    Args:
        bbox_coords (Dict[str,  np.ndarray): A dictionary containing
        bounding box coordinates for each key. Each value is a tuple representing
        the coordinates of the bounding box (xmin, ymin, xmax, ymax).

    Returns:
        Dict[str, np.ndarray]: A dictionary with keys corresponding to the input
        bounding box coordinates dictionary. The values are numpy arrays containing
        the mask representations for each bounding box.

    Raises:
        ValueError: If the input bbox_coords dictionary does not contain values of
        the correct format.
    """
    ground_truth_masks = {}
    for k in bbox_coords.keys():

        image_height, image_width, _channels = cv2.imread(
            os.path.join(dataset_images_path, k + ".png")
        ).shape

        mask_arr = np.zeros(shape=(image_height, image_width))
        bbox = bbox_coords[k]
        for box in bbox:
            xmin, ymin, xmax, ymax = (
                int(min(box[0], box[2])),
                int(min(box[1], box[3])),
                int(max(box[0], box[2])),
                int(max(box[1], box[3])),
            )
            mask_arr[ymin:ymax, xmin:xmax] = 255
        ground_truth_masks[k] = mask_arr

    return ground_truth_masks


def save_masks(
    ground_truth_masks_dict: Dict[str, np.ndarray],
    dataset_path: str,
    masks_dir=MASKS_DIR,
):
    """Save ground truth masks to a directory as PNG files.

    Args:
        ground_truth_masks_dict (Dict[str, np.ndarray]): A dictionary containing
        the ground truth masks for each image in the dataset, where keys are
        filenames and values are numpy arrays representing masks.
        dataset_path (str): The root directory of the dataset.
        masks_dir (str, optional): The directory within the dataset path to save
        the mask files. Defaults to 'masks'.

    Returns:
        None

    Raises:
        ValueError: If the `ground_truth_masks_dict` values are not array-like
        such that they can be saved to an image.
        FileNotFoundError: If `dataset_path` or `dataset_path` + `mask_dir` does
        not exist.
    """
    mask_dir_path = os.path.join(dataset_path, masks_dir)
    os.makedirs(mask_dir_path, exist_ok=True)

    for k in ground_truth_masks_dict.keys():
        out_file_name = k + ".png"
        out_file_path = os.path.join(mask_dir_path, out_file_name)
        cv2.imwrite(out_file_path, ground_truth_masks_dict[k])


if __name__ == "__main__":
    train_bbox_coords = get_split_bbox_coords(
        os.path.join(TRAIN_DATASET_PATH, LABELS_DIR),
        os.path.join(TRAIN_DATASET_PATH, IMAGES_DIR),
    )
    val_bbox_coords = get_split_bbox_coords(
        os.path.join(VAL_DATASET_PATH, LABELS_DIR),
        os.path.join(VAL_DATASET_PATH, IMAGES_DIR),
    )
    train_ground_truth_masks = convert_bbox_to_masks(
        train_bbox_coords, os.path.join(TRAIN_DATASET_PATH, IMAGES_DIR)
    )
    val_ground_truth_masks = convert_bbox_to_masks(
        val_bbox_coords, os.path.join(VAL_DATASET_PATH, IMAGES_DIR)
    )
    save_masks(train_ground_truth_masks, TRAIN_DATASET_PATH)
    save_masks(val_ground_truth_masks, VAL_DATASET_PATH)
