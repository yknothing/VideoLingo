import requests
from core.utils import load_key, except_handler


@except_handler("Failed to generate audio using Azure TTS", retry=3, delay=1)
def azure_tts(text: str, save_path: str) -> None:
    url = "https://api.302.ai/cognitiveservices/v1"

    API_KEY = load_key("azure_tts.api_key")
    voice = load_key("azure_tts.voice")

    payload = (
        f"""<speak version='1.0' xml:lang='zh-CN'><voice name='{voice}'>{text}</voice></speak>"""
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        "Content-Type": "application/ssml+xml",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to {save_path}")
    else:
        raise requests.exceptions.HTTPError(
            f"Azure TTS failed with status {response.status_code}: {response.text}"
        )


if __name__ == "__main__":
    azure_tts("Hi! Welcome to VideoLingo!", "test.wav")
