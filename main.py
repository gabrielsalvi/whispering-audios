import os
import sys
import time
import jiwer
import whisper
from whisper.normalizers import BasicTextNormalizer
from mutagen.mp3 import MP3

model_size = 'medium'
model = whisper.load_model(model_size)
normalizer = BasicTextNormalizer()
dictionary_keys = []

def main():
    open('results.txt', 'w').close()

    official_transcriptions = get_all_official_transcriptions('transcriptions')
    whisper_transcriptions = transcribe_all_audios_from_directory('audio')

    compare_transcriptions(official_transcriptions, whisper_transcriptions, dictionary_keys)

def transcribe_all_audios_from_directory(directory = 'audio'):
    transcriptions = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            key = remove_extension_from_filename(filename)
            filepath = os.path.join(root, filename)
            transcription = {}
            transcription['name'] = key
            start_time = time.time()
            whisper_transcription = model.transcribe(filepath, fp16=False)
            execution_time = time.time() - start_time
            _, minutes, seconds = get_duration(int(execution_time))
            transcription['time'] = str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
            transcription['text'] = normalizer(whisper_transcription['text']).strip()
            _, minutes, seconds = get_duration(int(MP3(filepath).info.length))
            transcription['length'] = str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
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
            transcription['num_of_words'] = len(transcription['text'].split())
            official_transcriptions[key] = transcription
            file.close()

    return official_transcriptions

def compare_transcriptions(official_transcriptions, whisper_transcriptions, keys):
    if len(official_transcriptions) != len(whisper_transcriptions):
        sys.exit('É necessário que o número de transcrições geradas pelo whisper seja igual ao número de transcrições oficiais!')

    for key in keys:
        officcial_transcription, whisper_transcription = get_transcription_value(official_transcriptions, whisper_transcriptions, key)

        measures = jiwer.compute_measures(officcial_transcription['text'], whisper_transcription['text'])
        
        whisper_transcription['misses'] = officcial_transcription['num_of_words'] - measures['hits']
        whisper_transcription['misses_percentage'] = round(whisper_transcription['misses'] * 100 / officcial_transcription['num_of_words'], 2)

        save_to_file(officcial_transcription, whisper_transcription)

def get_transcription_value(official_transcriptions, whisper_transcriptions, key):
    try:
        officcial_transcription = official_transcriptions[key]
        whisper_transcription = whisper_transcriptions[key]
        return officcial_transcription, whisper_transcription
    except:
        sys.exit('A transcrição do arquivo ' + key + ' não foi encontrada!')

def save_to_file(officcial_transcription, whisper_transcription):
    f = open('results.txt', 'a')

    if (os.stat(f.name).st_size == 0):
        f.write('Tamanho do modelo: ' + model_size.capitalize() + '\n')
        f.write('Idioma dos aúdios: Inglês' + '\n')

    f.write('\nNome do áudio: ' + officcial_transcription['name'] + '\n')
    f.write('Duração do áudio: ' + str(whisper_transcription['length']) + '\n')
    f.write('Transcrição oficial: ' + officcial_transcription['text'] + '\n')
    f.write('Transcrição do whisper: ' + whisper_transcription['text'] + '\n')
    f.write('Quantidade de palavras erradas: ' + str(whisper_transcription['misses']) + '\n')
    f.write('Percentual de palavras erradas: ' + str(whisper_transcription['misses_percentage']) + ' %\n')
    f.write('Tempo de execução: ' + whisper_transcription['time'] + '\n')

    f.close()

def remove_extension_from_filename(filename: str):
    return filename.split('.', 1)[0]

def get_duration(length):
    hours = length // 3600
    length %= 3600
    mins = length // 60
    length %= 60
    seconds = length
    return hours, mins, seconds

main()