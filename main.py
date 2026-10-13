# main.py
"""
Customer complaint classification workflow.

For each recorded complaint under audio/, this script:
  1. transcribes the audio            (whisper.py)
  2. builds an image prompt from it   (dalle.py)
  3. generates an image of the issue  (dalle.py)
  4. describes the image              (vision.py)
  5. annotates the reported issue     (vision.py)
  6. classifies the complaint         (gpt.py)

Every intermediate result is saved to output/<complaint_id>/ using the
file names required by the project rubric.
"""

import json
import os
import sys
from datetime import datetime

# Import functions from other modules
from whisper import transcribe_audio
from dalle import create_image_prompt, generate_image
from vision import describe_image, annotate_image
from gpt import classify_with_gpt
from config import OUTPUT_DIR, find_audio_files


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------

def save_text(folder, filename, content):
    """Write a text (or JSON-serializable) result into the complaint's output folder."""
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content, indent=2))
    print(f"  saved -> {os.path.relpath(path)}")
    return path


def format_image_description(vision_result):
    """Human-readable version of the vision model's output for image_description.txt."""
    x0, y0, x1, y1 = vision_result["bounding_box"]
    return (
        f"Description: {vision_result['description']}\n"
        f"Reported issue: {vision_result['issue']}\n"
        f"Issue location (fraction of image, x_min, y_min, x_max, y_max): "
        f"{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}\n"
    )


def format_classification(classification):
    """Human-readable version of the classifier's output for classification.txt."""
    text = (
        f"Category: {classification.get('category')}\n"
        f"Subcategory: {classification.get('subcategory')}\n"
        f"Reason: {classification.get('reason')}\n"
    )
    if classification.get("warning"):
        text += f"Warning: {classification['warning']}\n"
    return text


# ----------------------------------------------------------------------------
# Workflow
# ----------------------------------------------------------------------------

def process_complaint(complaint_id, audio_path):
    """
    Orchestrates the workflow for handling one customer complaint.

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
    folder = os.path.join(OUTPUT_DIR, complaint_id)
    os.makedirs(folder, exist_ok=True)
    print(f"\n=== {complaint_id} ({os.path.relpath(audio_path)}) ===")

    # 1. Transcribe the audio complaint.
    print("1. Transcribing audio complaint...")
    transcription = transcribe_audio(audio_path)
    save_text(folder, "transcription.txt", transcription)

    # 2. Create a prompt from the transcription.
    print("2. Creating image prompt...")
    image_prompt = create_image_prompt(transcription)
    save_text(folder, "image_prompt.txt", image_prompt)

    # 3. Generate an image based on the prompt.
    print("3. Generating image...")
    image_path = generate_image(image_prompt, os.path.join(folder, "generated_image.png"))
    print(f"  saved -> {os.path.relpath(image_path)}")

    # 4. Describe the generated image.
    print("4. Describing image...")
    vision_result = describe_image(image_path, transcription)
    save_text(folder, "image_description.txt", format_image_description(vision_result))

    # 5. Annotate the reported issue in the image.
    print("5. Annotating reported issue...")
    annotated_path = annotate_image(
        image_path,
        vision_result["issue"],
        vision_result["bounding_box"],
        os.path.join(folder, "annotated_image.png"),
    )
    print(f"  saved -> {os.path.relpath(annotated_path)}")

    # 6. Classify the complaint based on the image description.
    print("6. Classifying complaint...")
    classification = classify_with_gpt(vision_result["description"], transcription)
    save_text(folder, "classification.txt", format_classification(classification))

    # Store a machine-readable summary of every step.
    summary = {
        "complaint_id": complaint_id,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "audio_file": os.path.relpath(audio_path),
        "transcription": transcription,
        "image_prompt": image_prompt,
        "generated_image": os.path.relpath(image_path),
        "image_description": vision_result["description"],
        "reported_issue": vision_result["issue"],
        "bounding_box": vision_result["bounding_box"],
        "annotated_image": os.path.relpath(annotated_path),
        "category": classification.get("category"),
        "subcategory": classification.get("subcategory"),
        "reason": classification.get("reason"),
    }
    save_text(folder, "summary.json", summary)

    # Print the results.
    print("Result:")
    print(f"  Complaint:   {transcription}")
    print(f"  Issue:       {vision_result['issue']}")
    print(f"  Category:    {summary['category']} > {summary['subcategory']}")
    return summary


def main(only=None):
    """
    Process every complaint recording under audio/ (or just the ids given in `only`).
    Writes output/<complaint_id>/... for each one plus output/all_results.json.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    complaints = find_audio_files()
    if only:
        complaints = [c for c in complaints if c[0] in only]

    results = [process_complaint(cid, path) for cid, path in complaints]

    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Processed {len(results)} complaint(s). See output/all_results.json")
    return results


if __name__ == "__main__":
    # Usage: python main.py            -> all complaints
    #        python main.py customer1  -> just one
    main(only=sys.argv[1:] or None)
