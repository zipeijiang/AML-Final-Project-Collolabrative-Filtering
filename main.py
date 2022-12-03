import os
import json
import pickle
import numpy as np
import pandas as pd
import tqdm
import random
from util import data_preprocess, classifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

playlists = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/playlist.pk")
songs = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/tracks_df.pk")
sparseMatrix = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/sparse_matrix.pkl")
audioMatrix = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/audio_matrix.pkl")

class Model:
    def __init__(self, playlists, songs, matrix, case):
        self.playlists = playlists
        self.songs = songs
        self.matrix = matrix
        self.build(case)
    
    def build(self, case):
        if case=='sparse':
            self.build_sparse_classifier()
        elif case=='audio':
            self.build_audio_classifier()
        elif case=='mix':
            self.build_mixed_classifier()
        else:
            print('Invalid case input')

    def build_sparse_classifier(self):
        self.classifier = classifier.SparseClassifier(
            self.playlists, self.songs, self.matrix,
            metric='cosine', n_neighbors=30, load_exists=False) # TBC

    def build_audio_classifier(self):
        self.classifier = classifier.AudioClassifier(
            self.playlists, self.songs, self.matrix,
            metric='cosine', n_neighbors=30, load_exists=False) # TBC
        
    def build_mixed_classifier(self):
        self.classifier = classifier.MixedClassifier(
            self.playlists, self.songs, self.matrix,
            metric='cosine', n_neighbors=30, load_exists=False) # TBC    
        
    def predict(self, playlist, pid,  n_to_predict):
        return self.classifier.predict(playlist, pid, n_to_predict)
    
    def split_tracks(self, playlist, remove_fraction):  # fraction of songs to be removed
        tracks = np.array(playlist['tracks'])
        length = int(len(tracks)*remove_fraction)
        indexs = random.sample(range(len(tracks)), length)   # generate indexs to be removed
        s = set(indexs)
        left = [i for i in range(len(tracks)) if i not in s]
        removed_songs = tracks[indexs]
        kept_songs = tracks[left]
        return kept_songs, removed_songs
    
    def compute_accuracy_single(self, remove_fraction):
        playlist = self.playlists.iloc[random.randint(0,len(self.playlists)-1)]
        pid = playlist['pid']
        kept, removed = self.split_tracks(playlist, remove_fraction)
        removed = set(removed)
        pred = self.predict(kept, pid, 500)
        score = [1 if value in removed else 0 for value in pred]
        return sum(score)/len(removed)
        
    def evaluation(self, test_size, remove_fraction=0.5):
        scores = [self.compute_accuracy_single(remove_fraction) for i in tqdm.tqdm(range(test_size), position=0, leave=True)]
        accuracy = sum(scores) / len(scores)
        print(f"recommended {accuracy} of removed songs")

if __name__ == "__main__":
    #model = Model(playlists, songs, sparseMatrix, 'sparse')
    #model = Model(playlists, songs, audioMatrix, 'audio')
    model = Model(playlists, songs, (sparseMatrix, audioMatrix), 'mix')
    model.evaluation(30)
