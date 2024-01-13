import requests
import sys

# AI Voice Generator
# Define the available voices
voices = {
    "Alloy": "https://ai.mrc.fm/audio/alloy.mp3",
    "Echo": "https://ai.mrc.fm/audio/echo.mp3",
    "Fable": "https://ai.mrc.fm/audio/fable.mp3",
    "Onyx": "https://ai.mrc.fm/audio/onyx.mp3",
    "Nova": "https://ai.mrc.fm/audio/nova.mp3",
    "Shimmer": "https://ai.mrc.fm/audio/shimmer.mp3"
}

def validate_script(script):
    """
    Validate the length of the script.
    """
    return len(script.split()) <= 25

def select_voice():
    """
    Allow the user to select a voice.
    """
    print("Available voices:")
    for name in voices.keys():
        print(f"- {name}")

    voice = input("Choose a voice: ")
    if voice in voices:
        return voice
    else:
        print("Selected voice is not available. Please try again.")
        return select_voice()

def generate_voice_over(script, voice):
    """
    Generate a voice over from a given script and voice.
    """
    # In a real scenario, integrate with the actual API.
    # Here we simulate a successful API interaction.
    print(f"Generating voice over with the '{voice}' voice...")
    # Simulate a download link
    download_link = "https://example.com/download/voiceover.mp3"
    print("Voice over generated successfully. Download here:", download_link)

def main():
    print("Welcome to AI Voice Generator!")
    script = input("Please enter your script (max 25 words): ")

    if not validate_script(script):
        print("The script is too long. Please limit it to 25 words.")
        sys.exit()

    voice = select_voice()
    generate_voice_over(script, voice)

if __name__ == "__main__":
    main()
