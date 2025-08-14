"""
NAME
    benchmarking_script.py
DESCRIPTION
    This script offers a general functionality to separate audio into user specified window length chunks which overlap based on 
    user input. And uses SpeechBrain and Wa2Vec2 to benchmark on OptumRX, NVM and ITSS datasets. Other possible models from HuggingFace Hub could also be added,
    after following instructions under NOTES section.
    
EXAMPLE
    To Transcribe Using Microphone
    python inference_script_wav2vec_speechbrain_whisper.py --dataset=DATASET_NAME --model=MODEL_NAME --chunk-length=CHUNK_LENGTH --context-length=CONTEXT_LENGTH --chunk-transcription-directory-path=CHUNK_TRANSCRIPTION_DIRECTORY --machine_generated-transcription-path=MACHINE_TRANSCRIPTION_DIRECTORY --configuration-path=CONFIGURATION_PATH

    Arguments Definition:

        DATASET_NAME:
        - OptumRX
        - NVM
        - ITSS
        MODEL_NAME:
        - wav2vec2
        - speechbrain
        - whisper

        CHUNK_LENGTH(audio segment length in seconds):
        eg : 30 or 10 etc -> this is subject to experimentation, different values result in different WER/CER calculcations
        
        CONTEXT_LENGTH(overlap window length in seconds):
        eg : 0.1 or 1 -> this is subject to experimentation, different values result in different WER/CER calculcations
        
        TEMP_DIRECTORY ( directory to store <CHUNK_LENGTH> audio segments):
        eg : /data/atc_tenant/asr/User_folder/Temp
        
        CHUNK_TRANSCRIPTION_DIRECTORY ( directory to store transcriptions for <CHUNK_LENGTH> audio segments):
        eg : /data/atc_tenant/asr/User_folder/Split_Transcriptions

        MACHINE_TRANSCRIPTION_DIRECTORY ( directory to store machine generated transcriptions for selected DATASET_NAME ):
        eg: /data/atc_tenant/asr/User_folder/Machine_Generated 

        CONFIGURATION_PATH ( path to configuration file storing information about individual filenames for the OptumRX, NVM and ITSS datasets and their paths ):
        - /data/atc_tenant/asr/benchmarking_manifest_files/benchmarking_manifest.json

NOTES
    Additional models (custom model) from HuggingFace hub can be added to the script under the 'load_model' function. Also based on the model type a separate function to generate
    dataset needs to be created.
    The generate_dataset_custom_model(...) needs to be added under the generate_transcription function.
"""

import os
from os import listdir
from os.path import isfile, join
import sys
import shutil
import math
import argparse
import json
import torch
from pydub import AudioSegment 
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from pydub.utils import make_chunks
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, Audio
import whisper
import jiwer
from whisper.normalizers import EnglishTextNormalizer
#from transformers import WhisperForConditionalGeneration, WhisperProcessor
import whisper
import re
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import soundfile
from espnet2.bin.asr_inference import Speech2Text

def create_configuration(args):


    machine_generated_path = str(args.Machine_Transcription_Directory) + "-" + str(args.Dataset_Name) + "/"
    root_temp_path = str(args.Temp_Directory)
    chunk_audio_path = str(args.Temp_Directory) + "/Audio-Segments/"
    directory = str(args.Temp_Directory) + "/Segment-Transcriptions/"
    clear_folder_paths = [ directory, chunk_audio_path] 

    # get total files to transcribe

    print("Accessing DataFolder To Count Files To Transcribe : ")
    print_message = True
    audio_files_path, all_datasets = get_audio_files_path(args.Dataset_Name, args.CONFIGURATION_PATH, print_message)

    # if folder exists, give a warning that there are existing files in the folder and exit the program

    if(os.path.isdir(machine_generated_path)):


        print("Accessing Machine Transcription Directory : " + machine_generated_path)

        # re-create Temp folder structure, so that chunking can happen from start on the last file being processed before connection breaks

        if(os.path.isdir(root_temp_path)):
            delete_directory(root_temp_path)

        os.makedirs(chunk_audio_path)
        os.makedirs(directory)
        
        # get total files inside machine_generated_path

        files_transcribed = [f for f in listdir(machine_generated_path) if isfile(join(machine_generated_path,f)) and ".txt" in f]
        total_files_transcribed = len(files_transcribed)

        print(" >>> WARNING : There are total of : "+ str(total_files_transcribed) + " pre-existing transcribed file(s) in the " + str(machine_generated_path) + " directory <<< ")

        # check if the files to transcribe are already present in the machine generated path, then display a message and exit the code

        all_required_transcriptions = [ file.split("/")[-1].split(".")[0] + ".txt"  for file in audio_files_path ]
        total_required_transcriptions = len(all_required_transcriptions)

        counter = 0
        for file_to_transcribe in all_required_transcriptions:

            if file_to_transcribe in files_transcribed:
                counter = counter + 1

        if counter == total_required_transcriptions:

            print("All Required Transcriptions are present in : " + machine_generated_path)
            if(os.path.isdir(root_temp_path)):
                delete_directory(root_temp_path)
            sys.exit("Exiting Program")

    
    # else create the required directories
    else:
        os.makedirs(machine_generated_path)
        # if Temp Folder exists, delete it, and re-create it

        if(os.path.isdir(root_temp_path)):
            delete_directory(root_temp_path)
        
        os.makedirs(chunk_audio_path)
        os.makedirs(directory)
        
    return root_temp_path,chunk_audio_path, directory, clear_folder_paths, machine_generated_path

    

def get_audio_files_path(Dataset_Name, CONFIGURATION_PATH, print_message):
    
    # Reading from json file


    with open(CONFIGURATION_PATH, 'r') as openfile:
 
        
        json_object = json.load(openfile)


    all_datasets = list(json_object.keys())
    print("printing all_datasets",all_datasets)

    if Dataset_Name in list(json_object.keys()):

        audio_files_path = []


        audio_files = json_object[Dataset_Name]['audio_files']
        directory = json_object[Dataset_Name]['directory']

        for audio_file in audio_files:
    
    
            file_path = directory + audio_file
            audio_files_path.append(file_path)

        if print_message:
            print("Total Audio Files To Transcribe: " + str(len(audio_files_path)))
        audio_files_path.sort()

        return audio_files_path, all_datasets

    else:
        raise Exception("Unknown Dataset Name Entered -  Please Use Datasets Entered in Configuration File")

def generate_chunks(split_audio_path, single_audio_path, chunk_length_ms, overlapping_window_length):
    
    
    myaudio = AudioSegment.from_wav(single_audio_path) 
    
    file_index = single_audio_path.split("/")[-1].split(".")[0]
    
    
    chunk_length_ms = chunk_length_ms # pydub calculates in millisec 
    
    start_pos_chunk = 0 # inital positions of starting of a window
    end_pos_chunk = chunk_length_ms # inital positions of ending of a window
    overlapping_window_length = overlapping_window_length # window portion which will overlap with one another
    #chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of 30 sec 
    
    total_iterations = math.ceil(len(myaudio)/chunk_length_ms)
    
    for iteration in range(total_iterations):
        #print("starting : " + str(start_pos_chunk))
        #print("ending : " + str(end_pos_chunk))
        
        chunk = myaudio[start_pos_chunk : end_pos_chunk]
        start_pos_chunk = end_pos_chunk - overlapping_window_length
        end_pos_chunk = end_pos_chunk  + chunk_length_ms
    
        chunk_name = str(split_audio_path) + str(file_index) + "_" + str(iteration) + ".wav"
        #print(chunk_name)
        #print(type(chunk_name))
        
        if iteration < total_iterations:
            chunk.export(chunk_name, format="wav") 
    
      
    print(str(total_iterations) + " Overlapping Chunks created for file : " + str(file_index) + ".wav")
    print("exited generate_chunks")  

def assemble_files(file_path, Dataset_Name, all_datasets, extension):

    """file_path = path to all audio segments, or path to segment transcriptions"""

    file_path = [file_path + f for f in listdir(file_path) if isfile(join(file_path,f)) and extension in f]

        
    if ((Dataset_Name.casefold() == all_datasets[0].casefold()) and Dataset_Name=="OptumRX"): # for OptumRX
        file_path.sort(key=lambda x: [int(y) for y in x.split('/')[-1].split('.')[0].split('_')])
        return file_path

    elif (Dataset_Name=="NVM"): # for NVM
        file_path.sort(key=lambda x: [int(y) for y in x.split('/')[-1].split('.')[0].split('_')[-1]])
        return file_path

    elif (Dataset_Name=="ITSS"): # for ITSS
        file_path.sort(key=lambda x: [int(y) for y in x.split('/')[-1].split('.')[0].split('_')[-1]])
        return file_path
    else:
        raise Exception("Invalid Dataset Name Specified")

def generate_dataset(model_name, split_audio_path, Dataset_Name, all_datasets):
    
    # generated dataset for wav2vec2

    extension = ".wav"

    if model_name in ["wav2vec2"]:

        
        split_audio_path = assemble_files(split_audio_path, Dataset_Name, all_datasets, extension)

        audio_dataset = Dataset.from_dict({"audio": split_audio_path}).cast_column("audio", Audio(sampling_rate=16000))
        
        #print(split_audio_path)
        print("exited generate_dataset")
        return split_audio_path, audio_dataset
        
    # generated dataset for speechbrain

    if model_name in ["speechbrain"]:

        split_audio_path = assemble_files(split_audio_path, Dataset_Name, all_datasets, extension)
       
        print("exited generate_dataset")

        audio_dataset = None

        return split_audio_path, audio_dataset

    if model_name in ['whisper']:

        split_audio_path = assemble_files(split_audio_path, Dataset_Name, all_datasets, extension)

        audio_dataset = Dataset.from_dict({"audio": split_audio_path}).cast_column("audio", Audio(sampling_rate=16000))
        
        #print(split_audio_path)
        print("exited generate_dataset")
        return split_audio_path, audio_dataset
    
    if model_name in ["espnet"]:

        
        split_audio_path = assemble_files(split_audio_path, Dataset_Name, all_datasets, extension)

        audio_dataset = Dataset.from_dict({"audio": split_audio_path}).cast_column("audio", Audio(sampling_rate=16000))
        
        #print(split_audio_path)
        print("exited generate_dataset")
        return split_audio_path, audio_dataset


    # add more models and dataset generator code here, basically return both path to audio files and audio dataset , or just path to audio files

def load_model(Model_Name):  

    if Model_Name.casefold() in ["wav2vec2"]:

        torch.cuda.empty_cache()
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        wav2vec2 = wav2vec2.to('cuda')
        return wav2vec2, processor

    elif Model_Name.casefold() in ["speechbrain"]:

        asr_model_transformer = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech", run_opts={"device":"cuda:1"} )
        processor = None

        return asr_model_transformer,processor

        # add other model names here
    
    elif Model_Name in ["espnet"]:
        print("in espnet")

        asr_model_transformer = Speech2Text.from_pretrained("Yuekai Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave",
                                                            maxlenratio=0.0,
                                                            minlenratio=0.0,
                                                            beam_size=20,
                                                            ctc_weight=0.3,
                                                            lm_weight=0.5,
                                                            penalty=0.0,
                                                            nbest=1)
        processor = None

        return asr_model_transformer,processor

    else:

        raise Exception("Invalid Model Name Specified")

def generate_inference_speechbrain(asr_model_transformer, split_audio_path, directory):
    
    # the model and processor objects can be customized. All Hugging Face Models.
    
    torch.cuda.empty_cache()
     
    model_transcription = []
    
    for file_number in range(len(split_audio_path)):
        torch.cuda.empty_cache()
        filename = split_audio_path[file_number].split("/")[-1].split(".")[0]
        filepath = split_audio_path[file_number]
        
        model_transcription = (asr_model_transformer.transcribe_file(filepath))
        

        f = open(directory + filename + ".txt", "w")
        f.write(model_transcription)
        f.close()
        
    print("exited generate_inference")

def generate_inference_wav2vec2(model,processor, split_audio_path, audio_dataset, directory):
    
    # the model and processor objects can be customized. All Hugging Face Models.
    
    torch.cuda.empty_cache()
    
    for file_number in range(len(split_audio_path)):

        
        filename = split_audio_path[file_number].split("/")[-1].split(".")[0]
        inputs = processor(audio_dataset[file_number]["audio"]["array"], sample_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        model_transcription = processor.batch_decode(predicted_ids)
        torch.cuda.empty_cache()

        f = open(directory + filename + ".txt", "w")
        f.write(model_transcription[0])
        f.close()
        
    print("exited generate_inference")

def generate_inference_whisper(model, split_audio_path, directory):
    
    torch.cuda.empty_cache()
    
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    for file_number in range(len(split_audio_path)):
        torch.cuda.empty_cache()
        filename = split_audio_path[file_number].split("/")[-1].split(".")[0]
        audio_file_path = split_audio_path[file_number]
        
        result = model.transcribe(audio_file_path)
        
        result = result['text']
        
        transcription = emoji_pattern.sub(r'', result)

        f = open(directory + filename + ".txt", "w")
        f.write(transcription)
        f.close()
    print("exited generate_inference")

def generate_inference_espnet(asr_model_transformer, split_audio_path, directory):
    
    # the model and processor objects can be customized. All Hugging Face Models.
    
    torch.cuda.empty_cache()
     
    model_transcription = []
    
    for file_number in range(len(split_audio_path)):
        torch.cuda.empty_cache()
        filename = split_audio_path[file_number].split("/")[-1].split(".")[0]
        #print("split_audio_path", split_audio_path)
        #print("filename",filename)
        filepath = split_audio_path[file_number]
        #print("filepath",filepath)
        
        speech, rate = soundfile.read(filepath)
        nbests = asr_model_transformer(speech)

        text, *_ = nbests[0]
        model_transcription = text
        model_transcription = text
        my_new_string = model_transcription.replace("[noise]","").replace("[vocalized-noise]","").replace("[laughter]","")

        f = open(directory + filename + ".txt", "w")
        f.write(my_new_string)
        f.close()
        
    print("exited generate_inference")

def clear_directory(directory):
    

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    print("All Files deleted in : " + directory)

    print("exited clear_directory")

def clear_root(directory):
    

    for filename in os.listdir(directory):
        if ".wav" in filename:
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    print("All WAV Cache Files deleted in : " + directory)



def combine_inference(single_audio_path, machine_generated_path, directory, clear_folder_paths, Dataset_Name, all_datasets):
    
    
    extension = ".txt"

    split_transcriptions = assemble_files(directory, Dataset_Name, all_datasets, extension)


    all_transcriptions = []
    for file in split_transcriptions:

        f = open(file, "r")
        all_transcriptions.append(f.read())
    
    
    result = " ".join(all_transcriptions)
    
    file_index = single_audio_path.split("/")[-1].split(".")[0]
    
    f = open(machine_generated_path + file_index + ".txt", "w")
    f.write(result)
    f.close()
    
    print("All transcriptions combined for " +  file_index + " generated")
    
    
    for directory in clear_folder_paths:
        clear_directory(directory)


    current_working_directory = os.getcwd()

    clear_root(current_working_directory)
        
    print("exited combine_inference")



# deletes the Temp Folder after all transcriptions have been generated
def delete_directory(root_temp_path):
    try:
        shutil.rmtree(root_temp_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def generate_transcription(Dataset_Name, Model_Name, root_temp_path, chunk_audio_path, directory, clear_folder_paths, machine_generated_path, chunk_length_ms, overlapping_window_length_ms, CONFIGURATION_PATH):
    
    all_generated_transcriptions = [f for f in listdir(machine_generated_path) if isfile(join(machine_generated_path,f)) and ".txt" in f]
    print("generate_transcription")

    try:
        print_message = False
        audio_files_path, all_datasets = get_audio_files_path(Dataset_Name, CONFIGURATION_PATH, print_message)
        model, processor = load_model(Model_Name)
        

        for FILE_NO in range(len(audio_files_path)):
            
            # get file no : 
            single_audio_path = audio_files_path[FILE_NO]

            single_file_index = single_audio_path.split("/")[-1].split(".")[0]
        
            single_file_name = single_file_index + ".txt"
        

            # continue transcription from most recent file, incase connection breaks
            if single_file_name in all_generated_transcriptions:

                continue

            else:

                # create overlapping chunks of the file
                print("generating audio chunks for : " + single_audio_path)
                generate_chunks(chunk_audio_path, single_audio_path, chunk_length_ms, overlapping_window_length_ms,)

                # generate huggingface audio dataset in the same order the files were created

                model_name = Model_Name.casefold()

                if model_name in ["wav2vec2"]:
               
                    split_audio_path, audio_dataset = generate_dataset(model_name, chunk_audio_path, Dataset_Name, all_datasets)

                    # generated inference for all the chunks created for that specific file

                    generate_inference_wav2vec2(model,processor, split_audio_path, audio_dataset, directory)

                if model_name in ["speechbrain"]:

                    split_audio_path, audio_dataset = generate_dataset(model_name, chunk_audio_path, Dataset_Name, all_datasets)

                    # generated inference for all the chunks created for that specific file

                    generate_inference_speechbrain(model, split_audio_path, directory)


                if model_name in ["whisper"]:

                    split_audio_path, audio_dataset = generate_dataset(model_name, chunk_audio_path, Dataset_Name, all_datasets)
                    

                    # generated inference for all the chunks created for that specific file

                    generate_inference_whisper(model, split_audio_path, directory)
                    
                if model_name in ["espnet"]:
               
                    split_audio_path, audio_dataset = generate_dataset(model_name, chunk_audio_path, Dataset_Name, all_datasets)

                    # generated inference for all the chunks created for that specific file

                    generate_inference_espnet(model, split_audio_path, directory)


                # combine the inferences together and delete the chunks and individual transcriptions created
                combine_inference(single_audio_path, machine_generated_path, directory, clear_folder_paths, Dataset_Name, all_datasets)

                print("Total Files Processed : " + str(FILE_NO + 1) )

                print("######################################################################################")

        delete_directory(root_temp_path)
        print("All Transcriptions Generated")

    except:

        raise Exception("Incorrec Dataset Or Model Configuration Passed -  Please Use Dataset and Model Names in Configuration File")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description= "Continuous Streaming Transcription via Azure Services")

    parser.add_argument("--dataset", dest = "Dataset_Name", required = True, type = str, help = "Name of the dataset - audio files to transcribe" )

    parser.add_argument('--model', dest='Model_Name', required = True, type = str,help='Name of the model to use for transcription')

    parser.add_argument('--chunk-length', dest='CHUNK_LENGTH', required = True, type=int, help='Length of each audio segment')

    parser.add_argument('--context-length', dest='CONTEXT_LENGTH', required = True, type=float, help='Length of overlapping window between each audio segment')

    parser.add_argument('--temp-directory-path', dest='Temp_Directory', required = True, type=str, help='Path to Store Audio Chunks and Respective Transcriptions')

    parser.add_argument('--machine-generated-transcription-path', dest='Machine_Transcription_Directory', required = True, type=str, help='Path to Store Complete Transcriptions')

    parser.add_argument('--configuration-path', dest='CONFIGURATION_PATH', required = True, type=str, help='Path to the benchmarking_manfest JSON')

    args = parser.parse_args()

    print("User Selections : ")
    print(args)

    print("Creating Conifiguration for " + str(args.Dataset_Name))

    root_temp_path,chunk_audio_path, directory, clear_folder_paths, machine_generated_path = create_configuration(args)

    chunk_length_ms = args.CHUNK_LENGTH * 1000
    overlapping_window_length_ms = args.CONTEXT_LENGTH * 1000

    generate_transcription(args.Dataset_Name, args.Model_Name, root_temp_path, chunk_audio_path, directory, clear_folder_paths, machine_generated_path, chunk_length_ms, overlapping_window_length_ms, args.CONFIGURATION_PATH)
