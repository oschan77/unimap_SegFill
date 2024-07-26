# SegFill
SegFill is to separate the foreground from the background in an image, using a provided mask image (seed) for guidance.

In the seed image:
    - Red indicates the foreground.
    - Blue indicates the background.

## Example Usage
```
python src/segment_and_fill.py -o samples/pup.png -s samples/pup-seeds.png -f filled-pup.png
```