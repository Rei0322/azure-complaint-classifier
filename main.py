# main.py

import json
import os
from datetime import datetime

# Import functions from other modules
from whisper import transcribe_audio
from dalle import create_image_prompt, generate_image
from vision import describe_image, annotate_image
from gpt import classify_with_gpt
from config import OUTPUT_DIR


def save_text(filename, content):
    """Save an intermediate result to the output/ folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content, indent=2))
    print(f"  saved -> output/{filename}")
    return path


# Main function to orchestrate the workflow


def main(audio_path=None):
    """
    Orchestrates the workflow for handling customer complaints.

    Steps include:
    1. Transcribe the audio complaint.
    2. Create a prompt from the transcription.
    3. Generate an image representing the issue.
    4. Describe the generated image.
    5. Annotate the reported issue in the image.
    6. Classify the complaint into a category/subcategory pair.

    Returns:
    dict: Summary of every step's result.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Transcribing audio complaint...")
    transcription = transcribe_audio(audio_path)
    save_text("1_transcription.txt", transcription)

    print("2. Creating image prompt...")
    image_prompt = create_image_prompt(transcription)
    save_text("2_image_prompt.txt", image_prompt)

    print("3. Generating image...")
    image_path = generate_image(image_prompt)
    print(f"  saved -> output/{os.path.basename(image_path)}")

    print("4. Describing image...")
    vision_result = describe_image(image_path, transcription)
    save_text("4_image_description.json", vision_result)

    print("5. Annotating reported issue...")
    annotated_path = annotate_image(image_path, vision_result["issue"], vision_result["bounding_box"])
    print(f"  saved -> output/{os.path.basename(annotated_path)}")

    print("6. Classifying complaint...")
    classification = classify_with_gpt(vision_result["description"], transcription)
    save_text("6_classification.json", classification)

    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "transcription": transcription,
        "image_prompt": image_prompt,
        "generated_image": os.path.relpath(image_path),
        "image_description": vision_result["description"],
        "reported_issue": vision_result["issue"],
        "annotated_image": os.path.relpath(annotated_path),
        "category": classification.get("category"),
        "subcategory": classification.get("subcategory"),
        "reason": classification.get("reason"),
    }
    save_text("summary.json", summary)

    print("\nResult:")
    print(f"  Complaint:   {transcription}")
    print(f"  Issue:       {vision_result['issue']}")
    print(f"  Category:    {summary['category']} > {summary['subcategory']}")
    return summary


if __name__ == "__main__":
    main()
