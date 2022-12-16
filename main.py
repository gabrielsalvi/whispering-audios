import os
import sys
import time
import jiwer
import whisper
from whisper.normalizers import BasicTextNormalizer

model = whisper.load_model('medium')
normalizer = BasicTextNormalizer()
dictionary_keys = []

def main():
    official_transcriptions = get_all_official_transcriptions('transcriptions')
    whisper_transcriptions = transcribe_all_audios_from_directory('audio')

    compare_transcriptions(official_transcriptions, whisper_transcriptions, dictionary_keys)

def transcribe_all_audios_from_directory(directory = 'audio'):
    transcriptions = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            key = remove_extension_from_filename(filename)
            file = os.path.join(root, filename)
            transcription = {}
            transcription['name'] = key
            start_time = time.time()
            whisper_transcription = model.transcribe(file, fp16=False)
            transcription['time'] = time.time() - start_time
            transcription['text'] = normalizer(whisper_transcription['text']).strip()
            transcription['length'] = len(transcription['text'].split())
            transcriptions[key] = transcription

    return transcriptions;

def get_all_official_transcriptions(directory = 'transcriptions'):
    official_transcriptions = {}
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            key = remove_extension_from_filename(filename)
            dictionary_keys.append(key)
            file = open(os.path.join(root, filename))
            transcription = {}
            transcription['name'] = key
            transcription['text'] = normalizer(file.read()).strip()
            transcription['length'] = len(transcription['text'].split())
            official_transcriptions[key] = transcription
            file.close()

    return official_transcriptions

def compare_transcriptions(official_transcriptions, whisper_transcriptions, keys):
    if len(official_transcriptions) != len(whisper_transcriptions):
        sys.exit('É necessário que o número de transcrições geradas pelo whisper seja igual ao número de transcrições oficiais!')

    for key in keys:
        officcial_transcription, whisper_transcription = get_transcription_value(official_transcriptions, whisper_transcriptions, key)

def get_transcription_value(official_transcriptions, whisper_transcriptions, key):
    try:
        officcial_transcription = official_transcriptions[key]
        whisper_transcription = whisper_transcriptions[key]
        return officcial_transcription, whisper_transcription
    except:
        sys.exit('A transcrição do arquivo ' + key + ' não foi encontrada!')

def remove_extension_from_filename(filename: str):
    return filename.split('.', 1)[0]

main()