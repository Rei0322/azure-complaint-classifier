# gpt.py

import json

from config import get_client, GPT_DEPLOYMENT, GPT_API_VERSION, CATEGORIES_PATH

# Function to classify the customer complaint based on the image description


def load_categories():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_with_gpt(image_description, complaint=""):
    """
    Classifies the customer complaint into a category/subcategory based on the image description.

    Args:
    image_description (str): Description of the generated image from vision.py.
    complaint (str, optional): The transcribed complaint, for extra context.

    Returns:
    dict: {"category": str, "subcategory": str, "reason": str}
    """
    categories = load_categories()

    # Create a prompt that includes the image description and other relevant details.
    prompt = (
        "Classify this customer complaint into exactly one category and one subcategory "
        "from the catalog below. The subcategory must belong to the chosen category, and "
        "both must be copied exactly as written in the catalog.\n\n"
        f"Customer complaint: \"{complaint}\"\n"
        f"Image description: \"{image_description}\"\n\n"
        f"Catalog:\n{json.dumps(categories, indent=2)}\n\n"
        'Reply with JSON only: {"category": "...", "subcategory": "...", "reason": "one sentence"}'
    )

    # Call the GPT model to classify the complaint based on the prompt.
    client = get_client(GPT_API_VERSION)
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You classify e-commerce customer complaints."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    # Extract and return the classification result.
    result = json.loads(response.choices[0].message.content)

    # Check the answer is a real pair from the catalog.
    if result.get("subcategory") not in categories.get(result.get("category"), []):
        result["warning"] = "Model returned a category/subcategory pair not found in categories.json"
    return result


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    classification = classify_with_gpt(
        "A laptop on a desk with a large crack across the screen.",
        "My new laptop arrived with a cracked screen.",
    )
    print(json.dumps(classification, indent=2))
