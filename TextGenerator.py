from __future__ import absolute_import, division, print_function

import os
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *

def ask_for_int(input_sequence):
    user_input_value = 0
    correct_sequence_size = False
    while not correct_sequence_size:
        correct_sequence_size = True
        user_input_value = input(input_sequence)
        try:
            user_input_value = int(user_input_value)
        except:
            correct_sequence_size = False
            print("Erreur : La valeur donnee n'est pas un entier. Recommencez.")
    return user_input_value

def ask_for_float(input_sequence):
    user_input_value = 0
    correct_sequence_size = False
    while not correct_sequence_size:
        correct_sequence_size = True
        user_input_value = input(input_sequence)
        try:
            user_input_value = float(user_input_value)
        except:
            correct_sequence_size = False
            print("Erreur : La valeur donnee n'est pas un reel. Recommencez.")
    return user_input_value

def ask_for_yes_no(input_sequence):
    user_input_value = 0
    correct_sequence_size = False
    while True:
        user_input_value = input(input_sequence + " (y : yes, n : no)")
        if(user_input_value == 'y' || user_input_value == "yes") :
            return True
        elif (user_input_value == 'n' || user_input_value == "no") :
            return False
        else 
            print("Erreur : La valeur donnee n'est ni 'y', ni 'n'. Recommencez.")

def generateTextFrom(text_source_filename, epoch_count = -1, max_sequence_length = 20)

    if not os.path.isfile(text_source_filename):
        print(text_source_filename + " : le fichier n'existe pas. Sortie...")
            return

    X, Y, char_idx = \
        textfile_to_semi_redundant_sequences(path, seq_maxlen=max_sequence_length, redun_step=3)

    g = tflearn.input_data(shape=[None, max_sequence_length, len(char_idx)])
    # Premier LSTM
    g = tflearn.lstm(g, 512, return_seq=True)
    # Taux de Dropout du 1er LSTM
    g = tflearn.dropout(g, 0.5)
    # Second LSTM
    g = tflearn.lstm(g, 512)
    # Taux de Dropout du 2nd LSTM
    g = tflearn.dropout(g, 0.5)
    
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=max_sequence_length,
                                  clip_gradients=5.0,
                                  checkpoint_path=text_source_filename + "_model")

    #training
    if(epoch_count < 1) :
            epoch_count = ask_for_int("Nombre d'epochs ?")

    print("Demarrage de l'entrainement avec " + epoch_count + " epochs a faire.")
    for i in range(epoch_count):
        seed = random_sequence_from_textfile(path, max_sequence_length)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id=text_source_filename+'gen')
        print("Epoch n°" + i + "/" + epoch_count + "terminee.")

    print("Entrainement termine. La generation du texte va commencer.")

    pursue = True
    while pursue:
            generated_text_length = ask_for_int("Taille sequence finale ? ex : 150")
            generated_text_temperature = ask_for_float("Indice de nouveauté ? ex 1.25")
            
            seed = random_sequence_from_textfile(path, max_sequence_length)
            
            print(m.generate(generated_text_length, temperature=generated_text_temperature, seq_seed=seed))
            
            pursue = ask_for_yes_no("Continuer ?")
