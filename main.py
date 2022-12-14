import whisper
import os

model = whisper.load_model('base')

def transcribe_all_audios_from_directory(directory = 'audio'):
    transcriptions = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file = os.path.join(root, filename)
            transcription = model.transcribe(file, fp16=False)
            transcriptions[filename] = transcription['text']

    return transcriptions;

def get_all_official_transcriptions(directory = 'transcriptions'):
    official_transcriptions = {}
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file = open(os.path.join(root, filename))
            transcription = file.read()
            official_transcriptions[filename] = transcription
            file.close()
            
    return official_transcriptions

official_transcriptions = get_all_official_transcriptions()
whisper_transcriptions = transcribe_all_audios_from_directory('audio')