"""Small, labelled training augmentation for mixed-size plate lettering."""
import re


def mixed_font_plate(text, rng):
    """Render only a training registration, with a smaller prefix and large suffix.

    Uses OpenCV's built-in stroke fonts. The image is synthetic training material,
    not reconstructed camera evidence. No external fonts or weights are needed.
    """
    import cv2
    import numpy as np
    if not re.fullmatch(r"[A-Z0-9]{6,15}", text):
        raise ValueError("Synthetic training requires a complete ASCII registration")
    match = re.fullmatch(r"(.+?)(\d{3,4})", text)
    prefix, suffix = (match[1], match[2]) if match else (text[:4], text[4:])
    height = 64
    thickness = int(rng.integers(1, 3))
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX)
    font = int(rng.choice(fonts)) | (cv2.FONT_ITALIC if rng.random() < .5 else 0)
    base_height = cv2.getTextSize("H", font, 1., thickness)[0][1]
    prefix_scale = float(rng.uniform(20, 32)) / base_height
    suffix_scale = float(rng.uniform(38, 49)) / base_height
    prefix_width = cv2.getTextSize(prefix, font, prefix_scale, thickness)[0][0]
    suffix_width = cv2.getTextSize(suffix, font, suffix_scale, thickness)[0][0]
    gap = int(rng.integers(2, 8))
    background = (25, 195, 240) if rng.random() < .5 else (235, 235, 235)
    image = np.full((height, prefix_width + suffix_width + gap + 18, 3), background, np.uint8)
    baseline = height - 10
    cv2.putText(image, prefix, (8, baseline), font, prefix_scale, (15, 15, 15), thickness, cv2.LINE_AA)
    cv2.putText(image, suffix, (8 + prefix_width + gap, baseline), font, suffix_scale,
                (15, 15, 15), thickness, cv2.LINE_AA)
    if rng.random() < .3:
        image = cv2.GaussianBlur(image, (3, 3), .45)
    return image
