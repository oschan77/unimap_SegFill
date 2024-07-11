import argparse
import os

import cv2
import numpy as np

from GraphCutSegmentation import GraphCutSegmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original_img_path", type=str, required=False)
    parser.add_argument("-s", "--seeds_img_path", type=str, required=False)
    parser.add_argument("-f", "--filled_img_path", type=str, required=False)
    args = parser.parse_args()

    original_img_path = args.original_img_path
    seeds_img_path = args.seeds_img_path
    filled_img_path = args.filled_img_path
    obj_mask_path = "object-mask.png"
    filled_color = (120, 120, 120)  # BGR

    segmenter = GraphCutSegmentation()
    obj_mask = segmenter.get_object_mask(original_img_path, seeds_img_path)
    bkg_mask = ~obj_mask
    original_img_arr = segmenter.load_image(original_img_path, convert_to_bw=False)
    segmenter.save_masked_img(original_img_arr.copy(), obj_mask, obj_mask_path)

    image = cv2.imread(obj_mask_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel")
    alpha_channel = image[:, :, 3]
    background = np.full(
        (image.shape[0], image.shape[1], 3), filled_color, dtype=np.uint8
    )
    rgb_image = image[:, :, :3]
    alpha_norm = alpha_channel / 255.0
    alpha_norm = alpha_norm[..., np.newaxis]
    result = cv2.convertScaleAbs(rgb_image * alpha_norm + background * (1 - alpha_norm))

    cv2.imwrite(filled_img_path, result)
    os.remove(obj_mask_path)
    print(f">>> Output saved as {filled_img_path}")
