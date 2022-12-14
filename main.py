import whisper
import os

model = whisper.load_model('base')
 
def transcribe_all_audios_from_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file = os.path.join(root, filename)
            result = model.transcribe(file, fp16=False)
            print(result["text"] + "\n")

transcribe_all_audios_from_directory('audio')