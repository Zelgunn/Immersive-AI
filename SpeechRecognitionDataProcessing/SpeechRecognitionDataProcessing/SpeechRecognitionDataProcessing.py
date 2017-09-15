from SpeechMFCCPreprocessData import SpeechMFCCPreprocessData

import os
import gc
from tqdm import tqdm
import numpy as np
import math

def process_all_in_directory(dirpath:str, frame_rate = 16000, check_for_existing_npy_file = False):
    if not os.path.isdir(dirpath):
        print(dirpath + " is not a valid directory. Aborting.")
        return

    if not os.path.isdir(dirpath):
        print(dirpath + " is not a valid directory. Aborting.")
        return

    # Subfunction for gathering all files names
    def walk_into_librispeech(dirpath : str):
        def get_times_files_in_dir(root : str):
            files_in_directory = os.listdir(root)

            base_file = None
            intro_file = None

            for file in files_in_directory:
                if file.endswith(".sents.seg.txt"):
                #if file.endswith(".seg.txt") and not file.endswith(".sents.seg.txt"):
                    if "intro" in file:
                        intro_file = file
                    else:
                        base_file = file

            assert (base_file != None), "Couldn't find xx-xx.sents.seg.txt inside " + dirpath
            return (base_file, intro_file)

        audio_files = []
        times_files = []
        times_files_intro = []

        for root, directories, filenames in os.walk(dirpath):
            for filename in filenames:
                if filename.endswith(".mp3"):
                    ## Adding mp3 file to list
                    audio_files.append(os.path.join(root, filename))
                    ## Looking for *.seg.txt next to mp3 file (intro.seg.txt too, if present)
                    (base_file, intro_file) = get_times_files_in_dir(root)
                    times_files.append(os.path.join(root, base_file))
                    if intro_file is not None:
                        times_files_intro.append(os.path.join(root, intro_file))

        return (audio_files, times_files, times_files_intro)

    # Subfunction to extract times from files
    def get_all_times(times_files : list, frame_rate = 16000):
        all_times = []
        for times_file in times_files:
            file = open(times_file)
            all_text = file.read()

            lines = str.split(all_text, '\n')
            times = []
            for line in lines:
                if line is '':
                    continue

                infos = str.split(line, ' ')
                if(len(infos) < 3):
                    continue

                start = int(float(infos[1]) * frame_rate)
                end = int(float(infos[2]) * frame_rate)

                times.append([start, end])
            all_times.append(times)
        return all_times

    print("Searching mp3 files in", dirpath)
    (audio_files, times_files, times_files_intro) = walk_into_librispeech(dirpath)

    files_count = len(audio_files) # should be equal to the number of time files
    print("Found", files_count, "mp3 files,", len(times_files), "alignement files (plus", len(times_files_intro),"introduction aligment files) in", dirpath)
    print("Processing alignement files...")
    all_times = get_all_times(times_files)
    print("Alignement files processed !")

    print("Writing preprocess order file...")
    process_order = ""
    for i in range(files_count):
        process_order += str(i) + ',' + audio_files[i] + ',' + times_files[i]
        if i != (files_count - 1) :
            process_order += '\n'
    process_order_file = open(os.path.join(dirpath, "process_order.txt"), 'w')
    process_order_file.write(process_order)
    process_order_file.close()
    print("Preprocess order file written")

    print("Starting main pre process loop ...")
    for i in tqdm(range(files_count)):
        audio_file = audio_files[i]
        times = all_times[i]

        if check_for_existing_npy_file:
            npy_filename = SpeechDataPreprocess.get_npy_filename(audio_file)
            if os.path.exists(npy_filename):
                continue

        speech_data = SpeechDataPreprocess(audio_file, times)
        speech_data.save_fbank_as_binary()
        gc.collect()
    print("Job done !")

def gather_dictionnary(dirpath : str):
    spoken_text_files = []

    print("Looking for spoken books files (*.sents.trans.txt)")
    for root, directories, filenames in os.walk(dirpath):
        for filename in filenames:
            if filename.endswith(".sents.trans.txt"):
                spoken_text_files.append(os.path.join(root,filename))
    print("Found", len(spoken_text_files), "spoken books files")
        
    dictionnary = []
    print("Building dictionnary ...")
    for i in tqdm(range(len(spoken_text_files))):
        filename = spoken_text_files[i]
        file = open(filename, "r")
        all_text = file.read()

        lines = all_text.split('\n')
        for line in lines:
            split_index = line.find(' ') + 1
            if split_index < 1:
                continue
            words_in_line = line[split_index:]
            words_in_line = words_in_line.split(' ')

            for word in words_in_line:
                if word.endswith("\'S"):
                    word = word[:-2]
                if word.endswith("\'"):
                    word = word[:-1]
                if not word in dictionnary:
                    dictionnary.append(word)
    print("Finished building dictionnary, found", len(dictionnary), "different words")
        
    output_string = ""
    for i in tqdm(range(len(dictionnary))):
        output_string += str(i) + ' ' + dictionnary[i]
        if i < (len(dictionnary) - 1):
            output_string += '\n'

    print("Saving dictionnary under", os.path.join(dirpath, "dictionnary.txt"))

    output_file = open(os.path.join(dirpath, "dictionnary.txt"), 'w')
    output_file.write(output_string)        
    output_file.close()

#def dynamic_bucketing_preproc(dirpath : str):
#    # NPY arrays
#    mfcc_files = []

#    for root, directories, filenames in os.walk(dirpath):
#        for filename in filenames:
#            if filename.endswith(".npy"):
#                mfcc_files.append(os.path.join(root, filename))

#    lenghts_dict = dict()
#    for i in tqdm(range(len(mfcc_files))):
#        mfcc_file = mfcc_files[i]
#        mfcc_data = np.load(mfcc_file)

#        for sentence_mfcc_data in mfcc_data:
#            sentence_lenght = len(sentence_mfcc_data)
#            if sentence_lenght in lenghts_dict:
#                lenghts_dict[sentence_lenght] += 1
#            else:
#                lenghts_dict[sentence_lenght] = 1

#    lenghts_count = len(lenghts_dict)
#    lenghts = list(lenghts_dict)

#    output_string = ""
#    for i in range(lenghts_count):
#        lenght = lenghts[i]
#        output_string += str(lenght) + "," + str(lenghts_dict[lenght])
#        if (i + 1) < lenghts_count:
#            output_string += "\n"

#    output_file = open(os.path.join(dirpath, "lenghts.txt"), 'w')
#    output_file.write(output_string)        
#    output_file.close()


    # Sentences
#SpeechDataPreprocess.process_all_in_directory(r"F:\LibriSpeech\mp3", check_for_existing_npy_file = True)
#SpeechDataPreprocess.gather_dictionnary(r"F:\LibriSpeech\mp3")
#dynamic_bucketing_preproc(r"F:\LibriSpeech\mp3")

def pad_batches(dirpath : str):
  # NPY arrays
  mfcc_files = []

  for root, directories, filenames in os.walk(dirpath):
      for filename in filenames:
          if filename.endswith(".npy"):
              mfcc_files.append(os.path.join(root, filename))

  lenghts_dict = dict()
  for i in tqdm(range(len(mfcc_files))):
      mfcc_file = mfcc_files[i]
      mfcc_data = np.load(mfcc_file)

      for sentence_mfcc_data in mfcc_data:
          sentence_lenght = len(sentence_mfcc_data)
          if sentence_lenght in lenghts_dict:
              lenghts_dict[sentence_lenght] += 1
          else:
              lenghts_dict[sentence_lenght] = 1

  lenghts_count = len(lenghts_dict)
  lenghts = list(lenghts_dict)

  output_string = ""
  for i in range(lenghts_count):
      lenght = lenghts[i]
      output_string += str(lenght) + "," + str(lenghts_dict[lenght])
      if (i + 1) < lenghts_count:
          output_string += "\n"

  output_file = open(os.path.join(dirpath, "lenghts.txt"), 'w')
  output_file.write(output_string)        
  output_file.close()

pad_batches(r"D:\tmp\LibriSpeech\batches")