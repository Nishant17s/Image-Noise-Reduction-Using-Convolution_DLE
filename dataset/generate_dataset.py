"""
Dataset Generator Script
Creates clean reference test images for convolution-based noise reduction.
Includes high-frequency patterns, edges, gradients, and textures to evaluate
filter smoothing vs. edge preservation.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_synthetic_test_image(output_path: str, size: int = 512):
    """
    Creates a rich benchmark test image with diverse spatial frequencies,
    sharp boundaries, gradients, and shapes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.new("RGB", (size, size), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # 1. Background gradient
    for y in range(size):
        r = int(220 - (y / size) * 80)
        g = int(225 - (y / size) * 70)
        b = int(240 - (y / size) * 50)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # 2. Geometric Shapes (sharp edges)
    # Red circle
    draw.ellipse([40, 40, 160, 160], fill=(220, 50, 40), outline=(120, 20, 20), width=3)
    # Blue rectangle
    draw.rectangle([200, 50, 340, 150], fill=(40, 110, 210), outline=(20, 60, 130), width=3)
    # Emerald polygon
    draw.polygon([(420, 150), (370, 50), (470, 50)], fill=(30, 175, 110), outline=(15, 90, 50))

    # 3. Checkerboard high-frequency patch
    cb_size = 8
    cb_x, cb_y = 40, 200
    for i in range(16):
        for j in range(16):
            c = 255 if (i + j) % 2 == 0 else 20
            draw.rectangle(
                [cb_x + i * cb_size, cb_y + j * cb_size, cb_x + (i + 1) * cb_size, cb_y + (j + 1) * cb_size],
                fill=(c, c, c)
            )

    # 4. Concentric circular rings (frequency sweep / Siemens star concept)
    cx, cy = 270, 264
    for radius in range(55, 5, -8):
        c = int(255 * ((radius // 8) % 2))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(c, c, 180), width=3)

    # 5. Fine line grid (varying orientations)
    grid_x, grid_y = 360, 200
    for k in range(0, 120, 10):
        draw.line([(grid_x + k, grid_y), (grid_x + 120, grid_y + k)], fill=(50, 50, 50), width=2)
        draw.line([(grid_x, grid_y + k), (grid_x + k, grid_y + 120)], fill=(80, 80, 120), width=2)

    # 6. Text labels
    # Using default font
    draw.rectangle([40, 360, 470, 460], fill=(255, 255, 255), outline=(180, 180, 190), width=2)
    draw.text((60, 375), "IMAGE NOISE REDUCTION USING CONVOLUTION", fill=(20, 20, 30))
    draw.text((60, 400), "Deep Learning Essentials - Assignment II", fill=(70, 70, 90))
    draw.text((60, 425), "Evaluation: Average vs Gaussian Filters (PSNR / MSE / SSIM)", fill=(100, 100, 120))

    img.save(output_path, "PNG")
    print(f"Generated benchmark test image saved to: {output_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(out_dir, "sample_image.png")
    create_synthetic_test_image(sample_path)
