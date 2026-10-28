"""
Utility Script: clean_image.py
Converts a car image with a white or fake-checkerboard background into a true transparent PNG.

Usage:
    python clean_image.py input_car.png output_car.png
    python clean_image.py input_car.png (overwrites input_car.png as transparent PNG)
"""

import sys
from PIL import Image

def remove_background(input_path, output_path=None, tolerance=25):
    if output_path is None:
        output_path = input_path

    img = Image.open(input_path).convert("RGBA")
    data = img.getdata()

    new_data = []
    # Identify typical white / light-grey checkerboard colors:
    # White: (255, 255, 255), Light Grey: (204, 204, 204) to (245, 245, 245)
    for item in data:
        r, g, b, a = item
        # If pixels are near white or grey checkerboard pattern (where r ~= g ~= b and brightness > 190)
        is_grey_or_white = (
            abs(r - g) < tolerance and
            abs(g - b) < tolerance and
            abs(r - b) < tolerance and
            r > 190
        )
        if is_grey_or_white:
            new_data.append((255, 255, 255, 0))  # Full transparency
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(output_path, "PNG")
    print(f"Cleaned transparent PNG saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_image.py <image_path> [output_path]")
    else:
        in_p = sys.argv[1]
        out_p = sys.argv[2] if len(sys.argv) > 2 else in_p
        remove_background(in_p, out_p)
