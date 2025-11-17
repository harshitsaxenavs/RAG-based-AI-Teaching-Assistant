import os
import json
import whisper

audio_folder = "audios"

# Load  Whisper model
model = whisper.load_model("small")  

for file in os.listdir(audio_folder):
    if file.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, file)

        print(f" Transcribing: {file}")
        
        result = model.transcribe(
            audio_path,
            language="hi",
            task="translate",
            fp16=False,          
            beam_size=1,         
            temperature=0,     
            word_timestamps=False
        )

        # Extract chunks 
        chunks = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ]

        # Save JSON output
        json_name = file.replace(".mp3", ".json")
        with open(json_name, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f" Saved transcription → {json_name}\n")

print(" All MP3 files transcribed!")
