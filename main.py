import whisper
import os

model = whisper.load_model('base')
dictionary_keys = []

def main():
    official_transcriptions = get_all_official_transcriptions('transcriptions')
    whisper_transcriptions = transcribe_all_audios_from_directory('audio')

def transcribe_all_audios_from_directory(directory = 'audio'):
    transcriptions = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            key = remove_extension_from_filename(filename)
            file = os.path.join(root, filename)
            transcription = model.transcribe(file, fp16=False)
            transcriptions[key] = clean_transcription(transcription['text'])

    return transcriptions;

def get_all_official_transcriptions(directory = 'transcriptions'):
    official_transcriptions = {}
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            key = remove_extension_from_filename(filename)
            dictionary_keys.append(key)
            file = open(os.path.join(root, filename))
            transcription = file.read()
            official_transcriptions[key] = clean_transcription(transcription)
            file.close()

    return official_transcriptions

def remove_extension_from_filename(filename: str):
    return filename.split('.', 1)[0]

def clean_transcription(transcription: str):
    return transcription.replace('\n', ' ').replace('\r', ' ').strip()

main()