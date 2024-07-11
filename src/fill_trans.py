import cv2
import numpy as np

image = cv2.imread("bird-object-mask_alpha.png", cv2.IMREAD_UNCHANGED)

if image.shape[2] != 4:
    raise ValueError("Image does not have an alpha channel")

background_color = (120, 120, 120)  # BGR

alpha_channel = image[:, :, 3]

background = np.full(
    (image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8
)

rgb_image = image[:, :, :3]

alpha_norm = alpha_channel / 255.0
alpha_norm = alpha_norm[..., np.newaxis]

result = cv2.convertScaleAbs(rgb_image * alpha_norm + background * (1 - alpha_norm))

cv2.imwrite("filled.png", result)
