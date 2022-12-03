import os
import json
import pickle
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import dok_matrix
from sklearn.preprocessing import StandardScaler

FILES = os.listdir("C:/Users/14809/Desktop/W4995AML/Final-Project/data")
FILES.sort(key = lambda x: int(x.split('.')[2].split('-')[0]))

# not used
def load_data_to_df(num_files):
    
    track_uris = dict() # store all unique uris
    playlists_raw = [] # store uris and other info of each 
    tracks_raw = []
    
    files_to_read = FILES[:num_files]
    for file in tqdm.tqdm(files_to_read):
        data = json.load(open("C:/Users/14809/Desktop/W4995AML/Final-Project/data/"+file))
        playlists = data['playlists']  # 1k playlists
                
        for playlist in playlists:  
            tracks = playlist['tracks']
            for track in tracks:
                track_uri = track['track_uri'] # uri uniquely identifies a track
                if track_uri not in track_uris:
                    index = len(track_uris)
                    track_uris[track_uri] = index
                    tracks_raw.append(track)
                    #print(track_uris[track_uri] )
                track['track_uri'] = track_uris[track_uri] 
                
            playlist['tracks'] = [track['track_uri'] for track in tracks] # only want the uri
            playlists_raw.append(playlist)
        
    df_playlists = pd.DataFrame(playlists_raw)
    df_playlists.set_index('pid')
        
    df_tracks = pd.DataFrame(tracks_raw)
    df_tracks.set_index('track_uri')

    return df_playlists, df_tracks

def read_pickle(path_o, name_in):
    import pickle
    tmp_data = pickle.load(open(path_o + name_in + ".pk", "rb"))
    return tmp_data 

def create_sparse_matrix(playlists, tracks):
    print("creating sparse matrix")
    tids = list(tracks.index)
    pids = list(playlists['pid'])
    #tid_to_index = dict()
    #for i, tid in enumerate(tids):
    #    tid_to_index[tid] = i
    
    m, n = len(pids), int(tracks.index[-1])
    sparse_matrix = dok_matrix((m,n), dtype=np.float32)
    for i in tqdm.tqdm(range(m)):
        pid = pids[i]
        tids = playlists.loc[pid]["tracks"]
        t = list(filter(lambda x: x < n, tids))
        #tids = [tid_to_index[tid] for tid in tids]
        sparse_matrix[pid, t] = 1 

    return sparse_matrix.tocsr()#, tid_to_index


def create_audio_matrix(playlists, tracks):
    print("creating audio matrix")
    audio_data = tracks[['energy','key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'Vader_compound']]
    pids = list(playlists['pid'])
    m, n = len(pids), len(list(audio_data))

    matrix = []
    for i in tqdm.tqdm(range(m)):
        vec = np.zeros(n)
        curr_tracks_r = playlists.loc[i]['tracks']
        curr_tracks = list(filter(lambda x: x in audio_data.index, curr_tracks_r))
        length = len(curr_tracks)
        for song in curr_tracks:
            vec += np.array(audio_data.loc[song])
        matrix.append(vec/length)
    return np.array(matrix)

def preprocess_and_save_data(num_to_read):
    #df_playlists, df_tracks = load_data_to_df(num_files)
    df_playlists = read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/", "playlist")[:num_to_read]
    df_tracks = read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/", "tracks_df")
    #sparse_matrix = create_sparse_matrix(df_playlists, df_tracks)
    #pickle.dump(sparse_matrix, open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/sparse_matrix.pkl", "wb"))

    audio_matrix = create_audio_matrix(df_playlists, df_tracks)
    pickle.dump(audio_matrix, open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/audio_matrix.pkl", "wb"))