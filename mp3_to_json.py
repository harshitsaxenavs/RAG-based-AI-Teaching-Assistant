import whisper
import json
import os

# Load a faster model
model = whisper.load_model("medium")   

audios = os.listdir("audios")

for audio in audios:
    if "_" in audio and audio.endswith(".mp3"):

        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

        print(f" Transcribing:", number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # Build chunks list 
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        # Same outer structure
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        # Write JSON file
        os.makedirs("jsons", exist_ok=True)
        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

        print(f" Saved → jsons/{audio}.json\n")

print(" All done!")

