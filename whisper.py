# whisper.py

from config import get_client, find_audio_file, WHISPER_DEPLOYMENT, WHISPER_API_VERSION

# Function to transcribe customer audio complaints using the Whisper model


def transcribe_audio(audio_path=None):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.

    Args:
    audio_path (str, optional): Path to the audio file. Defaults to the first file in audio/.

    Returns:
    str: The transcribed text of the audio file.
    """
    # Load the audio file.
    audio_path = audio_path or find_audio_file()
    client = get_client(WHISPER_API_VERSION)

    # Call the Whisper model to transcribe the audio file.
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=WHISPER_DEPLOYMENT,
            file=audio_file,
        )

    # Extract the transcription and return it.
    return result.text.strip()


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    transcription = transcribe_audio()
    print(transcription)
