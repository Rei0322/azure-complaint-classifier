# vision.py

import base64
import json
import os
from mimetypes import guess_type

from PIL import Image, ImageDraw, ImageFont

from config import get_client, GPT_DEPLOYMENT, GPT_API_VERSION, OUTPUT_DIR

# Function to describe the generated image and annotate issues


def image_to_data_url(image_path):
    """Encode a local image as a base64 data URL for the vision model."""
    mime_type = guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _parse_json(text):
    """Pull a JSON object out of a model reply, even if wrapped in ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def describe_image(image_path, complaint=""):
    """
    Describes an image and identifies key visual elements related to the customer complaint.

    Args:
    image_path (str): Path to the generated image.
    complaint (str, optional): The transcribed complaint, for context.

    Returns:
    dict: {"description": str, "issue": str, "bounding_box": [x_min, y_min, x_max, y_max]}
          with the box in fractions (0-1) of the image width/height.
    """
    # Load the generated image.
    data_url = image_to_data_url(image_path)
    client = get_client(GPT_API_VERSION)

    prompt = (
        "You are inspecting a product image for a customer-service team.\n"
        f"Customer complaint: \"{complaint}\"\n\n"
        "1. Describe the image in 2-4 sentences: the product, its condition, and the setting.\n"
        "2. Name the specific reported issue visible in the image (e.g. 'cracked screen').\n"
        "3. Give a bounding box around that issue as fractions of the image size.\n\n"
        "Reply with JSON only, in this exact shape:\n"
        '{"description": "...", "issue": "...", "bounding_box": [x_min, y_min, x_max, y_max]}'
    )

    # Call the model to describe the image and identify key elements.
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a precise visual inspection assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=600,
        temperature=0,
    )

    # Extract the description and return it.
    return _parse_json(response.choices[0].message.content)


def _label_font(size):
    """Return a scalable TrueType font, falling back to Pillow's built-in font."""
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def annotate_image(image_path, issue, bounding_box, output_path=None):
    """
    Draws a labeled box around the reported issue and saves a new image.

    Returns:
    str: Path to the annotated image.
    """
    output_path = output_path or os.path.join(OUTPUT_DIR, "annotated_image.png")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Convert fractional box to pixels (and clamp to the image).
    x0, y0, x1, y1 = bounding_box
    box = [
        max(0, min(width, x0 * width)),
        max(0, min(height, y0 * height)),
        max(0, min(width, x1 * width)),
        max(0, min(height, y1 * height)),
    ]

    draw = ImageDraw.Draw(image)
    line_width = max(4, width // 200)
    draw.rectangle(box, outline="red", width=line_width)

    # Label sized to the image (about 3% of its height) so it stays readable.
    label = f"Issue: {issue}"
    pad = max(6, height // 150)
    size = max(16, height // 32)
    font = _label_font(size)
    # Shrink long labels until they fit across the image.
    while size > 12 and draw.textlength(label, font=font) > width - 4 * pad:
        size -= 2
        font = _label_font(size)
    label_height = draw.textbbox((0, 0), label, font=font)[3]
    above = box[1] - label_height - 2 * pad
    text_y = above if above > 0 else box[3] + pad
    text_x = min(box[0], max(0, width - draw.textlength(label, font=font) - 2 * pad))
    text_box = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle([text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad], fill="red")
    draw.text((text_x, text_y), label, fill="white", font=font)

    image.save(output_path)
    return output_path


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    test_image = os.path.join(OUTPUT_DIR, "generated_image.png")
    result = describe_image(test_image)
    print(json.dumps(result, indent=2))
    print(annotate_image(test_image, result["issue"], result["bounding_box"]))
