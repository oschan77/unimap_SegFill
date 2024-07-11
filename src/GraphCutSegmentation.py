from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from mincut_maxflow import SINK, SOURCE, GraphCutFastBFS, GraphCutFastRND
from parameters import DEFAULT_LAMBDA, DEFAULT_SIGMA


class GraphCutSegmentation:
    def __init__(self, sigma=DEFAULT_SIGMA, lambda_=DEFAULT_LAMBDA):
        self.sigma = sigma
        self.lambda_ = lambda_
        self.graph_cut = None

    def load_image(self, img_path, convert_to_bw=False):
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file '{img_path}' not found.")
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            raise Exception(f"Invalid image file '{img_path}' selected.")
        if convert_to_bw:
            img = img.convert("L")
        return np.asarray(img, dtype=np.uint8)

    def load_seeds(self, seeds_path):
        seeds_path = Path(seeds_path)
        if not seeds_path.exists():
            raise FileNotFoundError(f"Seeds file '{seeds_path}' not found.")
        seeds_img = Image.open(seeds_path)
        return np.asarray(seeds_img, dtype=np.uint8)

    def save_masked_img(self, img, mask, output_path):
        img[~mask] *= 0
        Image.fromarray(img).save(output_path)
        print(f">>> Output saved as {output_path}")

    def perform_cut(self, img_arr, seeds_arr, random_mode=False, animate_mode=False):
        if not np.any(seeds_arr):
            raise ValueError("Seeds array is empty. Can't perform graph cut.")

        cls_cut = GraphCutFastRND if random_mode else GraphCutFastBFS
        self.graph_cut = cls_cut(
            img_arr, seeds_arr, float(self.sigma), float(self.lambda_)
        )

        self.graph_cut.start_cut()
        while self.graph_cut.continue_cut():
            pass

        return self.graph_cut.output_array()

    def get_object_mask(self, img_path, seeds_path):
        img_arr = self.load_image(img_path, convert_to_bw=True)
        seeds_arr = self.load_seeds(seeds_path)
        result_arr = self.perform_cut(img_arr, seeds_arr)
        mask = self.graph_cut.TREE == SOURCE
        return mask

    def get_background_mask(self, img_path, seeds_path):
        img_arr = self.load_image(img_path, convert_to_bw=True)
        seeds_arr = self.load_seeds(seeds_path)
        result_arr = self.perform_cut(img_arr, seeds_arr)
        mask = self.graph_cut.TREE == SINK
        return mask


if __name__ == "__main__":
    original_img_path = "samples/bird.png"
    seeds_img_path = "samples/bird-seeds.png"
    obj_mask_path = "bird-object-mask.png"
    bkg_mask_path = "bird-background-mask.png"

    segmenter = GraphCutSegmentation()

    obj_mask = segmenter.get_object_mask(original_img_path, seeds_img_path)
    bkg_mask = ~obj_mask
    original_img_arr = segmenter.load_image(original_img_path, convert_to_bw=False)

    segmenter.save_masked_img(original_img_arr.copy(), obj_mask, obj_mask_path)
    segmenter.save_masked_img(original_img_arr.copy(), bkg_mask, bkg_mask_path)
