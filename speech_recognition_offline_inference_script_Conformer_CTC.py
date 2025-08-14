"""

DESCRIPTION
    This script offers  functionality to generate transcription of single channel OptumRX, NVM and ITSS datasets using the conformer ctc large model.
    
EXAMPLE
    To generate transcription use the following command:
    python offline_inference_script_Conformer_CTC.py --model_dir=<model_dir> --tokenizer_model_path=<tokenizer_model_path> --output_directory=<output_directory> --CONFIGURATION_PATH=<CONFIGURATION_PATH

    Arguments Definition:
    
        
    model_dir : Directory For Model Files - /data/atc_tenant/asr/Shekhar/nemo/asr_model_files/
    tokenizer_model_path : Directory For Tokenizer - /data/atc_tenant/asr/Shekhar/nemo/e06949b0b85a485e9f280ea6d19e5492_tokenizer.model
    output_directory : Directory To Save Output - /data/atc_tenant/asr/Shekhar/nemo/Batch_Nemo_NVM/
    CONFIGURATION_PATH : Path To The Configuration Files - /data/atc_tenant/asr/benchmarking_manifest_files/benchmarking_manifest.json


"""


import librosa
import argparse
import json
import audioop
import sys
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import random
import json
import soundfile as sf
import librosa

from torch.utils.data import DataLoader
import math
import abc

from tqdm import tqdm
import numpy as np
import torch
import tritonclient
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from ctcdecode import CTCBeamDecoder
import audio_metadata
from nemo.core.classes import IterableDataset
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor





def load_model_transcription_hyperparameters():
    stride = 4 # 8 for Citrinet
    window_size = 0.025
    feature_stride = 0.01
    sample_rate = 16000
    num_features = 80
    
    TOKEN_OFFSET = 100
    
    batch_size = 12

    
    # ids_to_tokens_func = tokenizer.ids_to_tokens
    # tokens_to_text_func = tokenizer.tokens_to_text

    beam_width = 100
    beam_alpha = 0.45
    beam_beta = 0
    
    return stride, window_size, feature_stride, sample_rate, num_features, TOKEN_OFFSET, beam_width, beam_alpha, beam_beta, batch_size



def load_model_artifacts(model_dir, tokenizer_model_path):
    
    
    stride, window_size, feature_stride, sample_rate, num_features, TOKEN_OFFSET, beam_width, beam_alpha, beam_beta, batch_size = load_model_transcription_hyperparameters()
    
    device = "cuda"
    pt_file = os.path.join(model_dir, "model.pt")
    lm_path = os.path.join(model_dir, "kenlm_6_medchat_rx_combined_conformer.bin")
    tokenizer_model_path = os.path.join(tokenizer_model_path)
    
    tokenizer = SentencePieceTokenizer(tokenizer_model_path, legacy=True)
    vocab = [chr(idx + TOKEN_OFFSET) for idx in range(tokenizer.tokenizer.get_piece_size())] + ["_"]
    
    
    preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=sample_rate, normalize='per_feature', window_size=window_size, window_stride=feature_stride, window='hann', features=num_features, n_fft=512, 
                                                 log=True, frame_splicing=1, dither=0.0, pad_to=0, pad_value=0.0)
    preprocessor.eval()
    
    
    model = torch.jit.load(pt_file, map_location=torch.device(device))
    model.eval()
    
    return model, lm_path, tokenizer, preprocessor, stride, window_size, feature_stride, sample_rate, num_features, TOKEN_OFFSET, beam_width, beam_alpha, beam_beta, batch_size, vocab



def load_CTCBeamDecoder(vocab, lm_path, beam_alpha, beam_beta, beam_width):
    
    decoder = CTCBeamDecoder(
    labels=vocab,
    model_path=lm_path,
    alpha=beam_alpha,
    beta=beam_beta,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=beam_width,
    num_processes=1,
    blank_id=len(vocab) - 1,
    log_probs_input=False)
    
    return decoder
    
 

class AudioChunkIterator():
    def __init__(self, samples, chunk_len_in_secs, sample_rate):
        print("******** In AudioChunkIterator *******")
        print(f"len samples: {len(samples)}")
        print(f"chunk_len_in_secs: {chunk_len_in_secs}")
        print(f"sample_rate: {sample_rate}")
        self._samples = samples
        self._chunk_len = chunk_len_in_secs * sample_rate
        self._start = 0
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False

        return chunk
    


sample_rate = 16000

def get_samples(audio_file, target_sr=sample_rate):
    with sf.SoundFile(audio_file, 'r') as f:
        #dtype = 'int16'
        dtype = 'float32'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
        # samples=samples.astype('float32')/32768
        samples = samples.transpose()
        return samples
    

def filter_signal(samp, sample_rate=8000, chunk_len_sec=0.2,
                  sample_width=2, min_noise_duration_sec=2.0,
                  max_noise_energy=20):
    """
    Filters background noise below RMS energy Threshold
    Calculates noise in 0.2 sec chunks up to 2 seconds and zeros samples
    If signal energy level in not exceeded in this 2 second window

    Args:
        samp(ndarray): array of float32 audio samples
        sample_rate(int): sample rate of audio signal
        chunk_len_sec(float): chunk len in seconds for RMS comparison
        sample width(int): number of bytes per sample, audioop rms requires this is to be 2
        min_noise_duration_sec(float): length in seconds of consecutive chunks below threshold to silence
        max_noise_energy(int): Root mean square energy threshold to remove signal lower than

    """

    samp_int = (samp * 32768).astype('int16')
    # process mini-chunks to detect (and zero out) silence
    chunk_len_frames = int(sample_rate * chunk_len_sec)

    silence_len = 0
    for start in range(0, len(samp_int), chunk_len_frames):
        data = samp_int[start:start + chunk_len_frames]
        energy = audioop.rms(np.ascontiguousarray(data), sample_width)
        data_len_sec = len(data) / sample_width / sample_rate  # just in case

        if energy < max_noise_energy:
            silence_len += data_len_sec
        else:
            silence_len = 0

        if silence_len > min_noise_duration_sec:      # minimum 2 seconds of silence for skipping
            samp[start:start + chunk_len_frames] = 0  # 0 silence in original array


def create_samples_list(samples):
    """

    Args:
        samples(Union[ndarray, tuple]): samples array if mono audio, tuple of samples if stereo

    Returns:
        samples(list): list of arrays of samples from audio file channels

    """

    num_channels = samples.shape[1] if len(samples.shape) > 1 else 1

    return [samples] if num_channels == 1 else [samples[:, i] for i in range(num_channels)]


def get_samples_from_file(audio_file, noise_reduction=False, target_sr=16000):
    """
    Reads Samples from wav and re-samples if necessary, calls noise reduction filter if flagged

    Args:
        audio_file(str): Path to audio file
        noise_reduction(bool): Flag to trigger noise reduction
        target_sr(int): Target sample rate of output audio file

    Returns:
        processed_samples_list(list): List of processed audio samples

    """

    with sf.SoundFile(audio_file, 'r') as f:

        sample_rate = f.samplerate
        samples = f.read(dtype='float32')
        samples_list = create_samples_list(samples)

        processed_samples_list = []
        for samples in samples_list:
            if noise_reduction:
                filter_signal(samples)

            if sample_rate != target_sr:
                samples = librosa.core.resample(samples, sample_rate, target_sr)
                samples = samples.transpose()

            processed_samples_list.append(samples)

        return processed_samples_list[0] if len(samples_list) == 1 else processed_samples_list



def get_metadata(audio_file):
    meta = audio_metadata.load(audio_file)
    output = dict(meta["streaminfo"])
    output = {key: output[key] for key in output if not key.startswith("_")}
    output["audio_format"] = str(output["audio_format"]).split(".")[-1]
    output["type"] = type(meta).__name__
    output = {key.replace("_", "-"): output[key] for key in output}

    return output


def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths


# softmax implementation in NumPy
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])



class AudioBuffersDataLayer(IterableDataset):

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32), torch.as_tensor(
            self.signal_shape[0], dtype=torch.int64)

    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1


# In[12]:


class ChunkBufferDecoderBase(abc.ABC):
    def __init__(self, model, tokenizer, preprocessor, batch_size, feature_stride, stride,
                 chunk_len_in_secs=1, buffer_len_in_secs=3):
        
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=batch_size, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.all_logits = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        self.start_delay_sec = (self.buffer_len - self.chunk_len) / 2

        assert (chunk_len_in_secs <= buffer_len_in_secs)

        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        print("tokens per chunk", self.n_tokens_per_chunk)
        self.blank_id = tokenizer.tokenizer.get_piece_size()

    def reset(self):
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.all_logits = []

    @torch.no_grad()
    def transcribe_buffers(self, buffers):
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self.get_preds()
        return self.decode_final()

    def get_preds(self):
        pass

    def decode_final(self):
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        probs = []
        for lgt in self.all_logits[:-1]:
            probs.append(softmax(lgt[len(lgt) - 1 - delay:len(lgt) - 1 - delay + self.n_tokens_per_chunk]))
        probs.append(softmax(self.all_logits[-1][len(self.all_logits[-1]) - 1 - delay:]))

        probs = np.concatenate(probs, axis=0)
        return self.beam_decode(probs)

    def beam_decode(self, probs):
        pass


class ChunkBufferDecoderBatch(ChunkBufferDecoderBase):

#     def __init__(self, model_infer, tokenizer, preprocessor, decoder, feature_stride, stride, batch_size,
#                  chunk_len_in_secs=1, buffer_len_in_secs=3):
    def __init__(self, model,tokenizer, preprocessor, decoder, feature_stride, stride, batch_size,
                 chunk_len_in_secs=1, buffer_len_in_secs=3):
        super(ChunkBufferDecoderBatch, self).__init__(model, tokenizer, preprocessor,
                                                      batch_size, feature_stride,
                                                      stride, chunk_len_in_secs,
                                                      buffer_len_in_secs)

#         self.model_infer = model_infer
        self.decoder = decoder

    def get_preds(self):
        for batch in iter(self.data_loader):
            audio_signal, audio_signal_len = batch
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_len,
            )
#             print(processed_signal)
#             logits = self.model_infer(processed_signal, processed_signal_length)
            logits = self.model(processed_signal.cuda(0), processed_signal_length.cuda(0)).cpu()
            print("******logits*****")
            print(len(logits))

            preds = logits.argmax(axis=-1)
            for lgt in logits:
                self.all_logits.append(lgt.numpy())
            for pred in preds:
                self.all_preds.append(pred.numpy())

    def beam_decode(self, probs):
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(
            torch.from_numpy(np.expand_dims(probs, axis=0)))
        beam_results = beam_results[0][0][:out_lens[0][0]].tolist()
        timesteps = timesteps[0][0][:out_lens[0][0]].tolist()
        tokens = self.tokenizer.ids_to_tokens(beam_results)
        pred_text = self.tokenizer.tokens_to_text(tokens)

        words = pred_text.split(" ")
        begin_timesteps = [int((t * self.model_stride_in_secs - self.start_delay_sec) * 1000) for idx, t in
                           enumerate(timesteps) if tokens[idx][0] == '▁']
        end_timesteps = [int((t * self.model_stride_in_secs - self.start_delay_sec) * 1000) for idx, t in
                         enumerate(timesteps) if idx == len(tokens) - 1 or tokens[idx + 1][0] == '▁']

        return pred_text, words, begin_timesteps, end_timesteps



def load_chunking_hyperparameters():
    # chunk_len_in_secs = 24
    # context_len_in_secs = 12
    chunk_len_in_secs = 6
    context_len_in_secs = 3
    sample_rate = 16000
    buffer_len_in_secs = chunk_len_in_secs + 2 * context_len_in_secs
    buffer_len = sample_rate*buffer_len_in_secs    
    print(chunk_len_in_secs, context_len_in_secs)
    chunk_len = sample_rate*chunk_len_in_secs
    
    
    return chunk_len_in_secs, context_len_in_secs, sample_rate, buffer_len_in_secs, buffer_len, chunk_len


# In[15]:


def load_ChunkBufferDecoderBatch(model, tokenizer,preprocessor, decoder, feature_stride,stride, batch_size,  chunk_len_in_secs, buffer_len_in_secs):

    asr_decoder = ChunkBufferDecoderBatch(model = model, tokenizer=tokenizer,
                                      preprocessor=preprocessor,
                                      decoder=decoder,
                                      feature_stride=feature_stride,
                                      stride=stride,
                                      batch_size=batch_size,
                                      chunk_len_in_secs=chunk_len_in_secs,
                                      buffer_len_in_secs=buffer_len_in_secs)
    
    
    return asr_decoder




def get_audio_file_path(CONFIGURATION_PATH):
    
    with open(CONFIGURATION_PATH, 'r') as openfile:
 
        
        json_object = json.load(openfile)
        

    key = [key for key in json_object.keys()][0]

    audio_files_path = []
    for file in json_object[key]['audio_files']:
        audio_files_path.append(json_object[key]['directory'] + file)
        
        
    return audio_files_path
    



def generate_transcription(audio_files_path, buffer_len, chunk_len, chunk_len_in_secs , sample_rate, asr_decoder, directory):
        
    
    all_audio_files = get_audio_file_path(CONFIGURATION_PATH)
    
    #all_required_transcriptions = [f.split(".")[0] + ".txt" for f in listdir(audi_files_path) if isfile(join(audi_files_path, f)) if ".wav" in f]
    
    
    print("Total WAV Files : " + str(len(all_audio_files)))
    
    counter = 1
    
    for audio_file in all_audio_files:
        
        
        filename = audio_file.split("/")[-1].split(".")[0]
        
        
        all_completed_transcriptions = [f for f in listdir(directory) if isfile(join(directory, f)) if ".txt" in f]
        
        
        if filename+".txt" in all_completed_transcriptions:
            print("Transcription Already Generated for " + str(filename) + ".txt")
            counter = counter + 1
            continue
            
        else:
        
        
            samples = get_samples_from_file(audio_file)

            chunk_reader_mono = AudioChunkIterator(samples=samples,
                                      chunk_len_in_secs=chunk_len_in_secs,
                                      sample_rate=sample_rate)


            sampbuffer = np.zeros([buffer_len], dtype=np.float32)
            print(f"*******len sampbuffer: {len(sampbuffer)}")
            buffer_list_mono = []

            for chunk in chunk_reader_mono:
                sampbuffer[:chunk_len] = sampbuffer[chunk_len:]
                sampbuffer[-chunk_len:] = chunk
                buffer_list_mono.append(np.array(sampbuffer))


            transcript_mono_batch, words_mono, begin_time_mono, end_time_mono = asr_decoder.transcribe_buffers(buffer_list_mono)


            f = open(directory + filename + ".txt", "w")
            f.write(transcript_mono_batch)
            f.close()

            print("File Transcribed : " + str(counter))

            counter = counter + 1

        
    print("All Transcriptions Generated for audio files in : " + str(audio_files_path))
    
    


def master_function(model_dir, tokenizer_model_path, CONFIGURATION_PATH, directory):
    
    
    model, lm_path, tokenizer, preprocessor, stride, window_size, feature_stride, sample_rate, num_features, TOKEN_OFFSET, beam_width, beam_alpha, beam_beta, batch_size, vocab = load_model_artifacts(model_dir, tokenizer_model_path)

    decoder = load_CTCBeamDecoder(vocab, lm_path, beam_alpha, beam_beta, beam_width)

    chunk_len_in_secs, context_len_in_secs, sample_rate, buffer_len_in_secs, buffer_len, chunk_len = load_chunking_hyperparameters()
    
    asr_decoder = load_ChunkBufferDecoderBatch(model, tokenizer,preprocessor, decoder, feature_stride,stride, batch_size,  chunk_len_in_secs, buffer_len_in_secs)
    
    generate_transcription(CONFIGURATION_PATH, buffer_len, chunk_len, chunk_len_in_secs , sample_rate, asr_decoder, directory)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= "Arguments for Transcription using NEMO")

    parser.add_argument("--model_dir", dest = "model_dir", required = True, type = str, help = "Directory For Model Files" )

    parser.add_argument('--tokenizer_model_path', dest='tokenizer_model_path', required = True, type = str,help= "Directory For Tokenizer")

    parser.add_argument('--output_directory', dest='output_directory', required = True, type=str, help='Directory To Save Output')

    parser.add_argument('--CONFIGURATION_PATH', dest='CONFIGURATION_PATH', required = True, type=str, help='Path To The Configuration Files')

    args = parser.parse_args()
    
    
    model_dir = str(args.model_dir) 
    tokenizer_model_path = str(args.tokenizer_model_path)
    output_directory = str(args.output_directory)
    CONFIGURATION_PATH = str(args.CONFIGURATION_PATH)

    master_function(model_dir, tokenizer_model_path, CONFIGURATION_PATH, output_directory)

