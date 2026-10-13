# dalle.py

import base64
import os

import requests

from config import get_client, DALLE_DEPLOYMENT, DALLE_API_VERSION, OUTPUT_DIR

# Function to generate an image representing the customer complaint


def create_image_prompt(transcription):
    """Turn the complaint transcription into a clear, photo-style image prompt."""
    return (
        "A realistic, well-lit product photo that clearly shows the problem described "
        f"in this customer complaint: \"{transcription}\". "
        "Focus on the damaged or defective product so the issue is easy to see. "
        "No text, no logos, no people's faces."
    )


def generate_image(prompt, output_path=None):
    """
    Generates an image based on a prompt using OpenAI's DALL-E model.

    Args:
    prompt (str): The image prompt built from the complaint.
    output_path (str, optional): Where to save the image. Defaults to output/generated_image.png.

    Returns:
    str: The path to the generated image.
    """
    client = get_client(DALLE_API_VERSION)
    output_path = output_path or os.path.join(OUTPUT_DIR, "generated_image.png")

    # Call the DALL-E model to generate an image based on the prompt.
    result = client.images.generate(
        model=DALLE_DEPLOYMENT,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    image = result.data[0]

    # Download the generated image and save it locally.
    # DALL-E 3 returns a URL; newer image models (gpt-image-1) return base64.
    if getattr(image, "b64_json", None):
        image_bytes = base64.b64decode(image.b64_json)
    else:
        response = requests.get(image.url, timeout=60)
        response.raise_for_status()
        image_bytes = response.content

    with open(output_path, "wb") as f:
        f.write(image_bytes)
    return output_path


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    test_prompt = create_image_prompt("My new laptop arrived with a cracked screen.")
    image_path = generate_image(test_prompt)
    print(f"Generated image saved at: {image_path}")
