import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SparseClassifier:
    def __init__(self, playlists, songs, matrix, metric='cosine', n_neighbors=60, load_exists=False): # may change metric and n
        self.playlists = playlists
        self.songs = songs
        self.matrix = matrix
        
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.build(load_exists)
    
    def build(self, load_exists):
        if load_exists:
            self.load_model()
        else:
            self.train_model()
        
    def train_model(self):
        self.model.fit(self.matrix)
        pickle.dump(self.model, open(f'C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel_sparse', 'wb'))
        
    def load_model(self):
        self.model = pickle.load(open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel_sparse", "rb"))
    
    def predict_by_neighbors(self, neighbor_songs, known_songs, n_to_predict):
        scores = dict()
        for i, songs in enumerate(neighbor_songs):
            for song in songs:
                if song not in scores:
                    if song not in known_songs:
                        scores[song] = 1/(i+1) # may change weight
                else:
                    scores[song] += 1/(i+1)
        sorted_scores = sorted(scores, key=scores.get, reverse=True)[:n_to_predict]
        return sorted_scores
    
    def predict(self, X, pid, n_to_predict):
        x_sparse = dok_matrix((1,2262195), dtype=np.float32)
        x_sparse[0, X] = 1
        x_vec = x_sparse.tocsr()
        
        x_neighbors = self.model.kneighbors(x_vec, return_distance=False)[0]
        neighbor_songs = [self.playlists.loc[i]['tracks'] for i in x_neighbors if (i in self.playlists.index and i != pid)]
        prediction = self.predict_by_neighbors(neighbor_songs, X, n_to_predict)
        #print(prediction)        
        return prediction

class AudioClassifier:
    def __init__(self, playlists, songs, matrix, metric='euclidean', n_neighbors=60, load_exists=False): # may change metric and n
        self.playlists = playlists
        self.songs = songs
        self.matrix = matrix
        
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.build(load_exists)
    
    def build(self, load_exists):
        if load_exists:
            self.load_model()
        else:
            self.train_model()
        
    def train_model(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.matrix)
        self.matrix = self.scaler.transform(self.matrix)
        self.model.fit(self.matrix)
        pickle.dump(self.model, open(f'C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel_audio', 'wb'))
        
        
    def load_model(self):
        self.model = pickle.load(open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel_audio", "rb"))
    
    def predict_by_neighbors(self, neighbor_songs, known_songs, n_to_predict):
        scores = dict()
        for i, songs in enumerate(neighbor_songs):
            for song in songs:
                if song not in scores:
                    if song not in known_songs:
                        scores[song] = 1/(i+1) # may change weight
                else:
                    scores[song] += 1/(i+1)
        sorted_scores = sorted(scores, key=scores.get, reverse=True)[:n_to_predict]
        return sorted_scores
    
    def predict(self, X, pid, n_to_predict):
        audio_data = np.array(self.songs.loc[X][['energy','key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'Vader_compound']])

        x_vec = [audio_data.sum(axis=0) / len(audio_data)]
        x_vec = self.scaler.transform(x_vec)
        x_neighbors = self.model.kneighbors(x_vec, return_distance=False)[0]
        neighbor_songs = [self.playlists.loc[i]['tracks'] for i in x_neighbors if (i in self.playlists.index and i != pid)]
        prediction = self.predict_by_neighbors(neighbor_songs, X, n_to_predict)

        return prediction

class MixedClassifier:
    def __init__(self, playlists, songs, matrixs, metric='euclidean', n_neighbors=60, load_exists=False): # may change metric and n
        self.playlists = playlists
        self.songs = songs
        self.sparse_matrix = matrixs[0]
        self.audio_matrix = matrixs[1]
        self.n_neighbors = int(n_neighbors/2)
        self.build(metric, self.n_neighbors, load_exists)

    def build(self, metric, n_neighbors, load_exists):
        self.classifier1 = SparseClassifier(self.playlists, self.songs, self.sparse_matrix, metric=metric, n_neighbors=n_neighbors, load_exists=load_exists)
        self.classifier2 = AudioClassifier(self.playlists, self.songs, self.audio_matrix, metric=metric, n_neighbors=n_neighbors, load_exists=load_exists) 

    def predict_by_neighbors(self, neighbor_songs, known_songs, n_to_predict):
        scores = dict()
        for i, songs in enumerate(neighbor_songs):
            for song in songs:
                if song not in scores:
                    if song not in known_songs:
                        scores[song] = 1/(i+1) # may change weight
                else:
                    scores[song] += 1/(i+1)
        sorted_scores = sorted(scores, key=scores.get, reverse=True)[:n_to_predict]
        return sorted_scores
    
    def predict(self, X, pid, n_to_predict):
        x_sparse = dok_matrix((1,2262195), dtype=np.float32)
        x_sparse[0, X] = 1
        x_vec1 = x_sparse.tocsr()

        audio_data = np.array(self.songs.loc[X][['energy','key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'Vader_compound']])
        x_vec2 = [audio_data.sum(axis=0) / len(audio_data)]
        x_vec2 = self.classifier2.scaler.transform(x_vec2)

        x_neighbors1 = self.classifier1.model.kneighbors(x_vec1, return_distance=False)[0]
        x_neighbors2 = self.classifier2.model.kneighbors(x_vec2, return_distance=False)[0]
        x_neighbors = []
        for i in range(self.n_neighbors):
            x_neighbors.append(x_neighbors2[i])
            x_neighbors.append(x_neighbors1[i])
        neighbor_songs = [self.playlists.loc[i]['tracks'] for i in x_neighbors if (i in self.playlists.index and i != pid)]
        prediction = self.predict_by_neighbors(neighbor_songs, X, n_to_predict)
             
        return prediction
