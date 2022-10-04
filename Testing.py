from operator import ge
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util


def generate():
    complete_feature_set=pd.read_hdf('Recommendation/feature_matrix.hdf')
    spotify_df = pd.read_csv('Recommendation/tracks.csv')
    spotify_df.drop(range(155475,523475),inplace=True)
    print('Done')

    client_id = '4613df3d871b47aea2b24f16d18256dc'
    client_secret= 'c95a03960e7f42d5bf549d9e7a0fffd5'

    scope = 'user-library-read'

    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://127.0.0.1:5000/')

    sp = spotipy.Spotify(auth=token)


    id_name = {}
    list_photo = {}

    for i in sp.current_user_playlists()['items']:
        print(i['uri'])
        id_name[i['name']] = i['uri'].split(':')[2]
        list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']

    print(id_name)

    print(list(id_name.keys()))

    first_playList=str(list(id_name.keys())[0])

    print(first_playList)


    def create_necessary_outputs(playlist_name,id_dic, df):
        """ 
        Pull songs from a specific playlist.

        Parameters: 
            playlist_name (str): name of the playlist you'd like to pull from the spotify API
            id_dic (dic): dictionary that maps playlist_name to playlist_id
            df (pandas dataframe): spotify datafram
            
        Returns: 
            playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
        """
        
        #generate playlist dataframe
        playlist = pd.DataFrame()
        playlist_name = playlist_name

        for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
            #print(i['track']['artists'][0]['name'])
            playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
            playlist.loc[ix, 'name'] = i['track']['name']
            playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
            playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
            playlist.loc[ix, 'date_added'] = i['added_at']

        playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
        
        playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
        
        return playlist


    playlist_WORKOUT = create_necessary_outputs(first_playList, id_name,spotify_df)



    def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
        """ 
        Summarize a user's playlist into a single vector

        Parameters: 
            complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
            playlist_df (pandas dataframe): playlist dataframe
            weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
            
        Returns: 
            playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
            complete_feature_set_nonplaylist (pandas dataframe): 
        """
        
        complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
        complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
        complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
        
        playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

        most_recent_date = playlist_feature_set.iloc[0,-1]
        
        for ix, row in playlist_feature_set.iterrows():
            playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
            
        playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
        
        playlist_feature_set_weighted = playlist_feature_set.copy()
        #print(playlist_feature_set_weighted.iloc[:,:-4].columns)
        playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
        playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
        #playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
        
        return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist

    complete_feature_set_playlist_vector_WORKOUT, complete_feature_set_nonplaylist_WORKOUT = generate_playlist_feature(complete_feature_set, playlist_WORKOUT, 1.09)

    print('ALL Done')

    print(complete_feature_set_playlist_vector_WORKOUT.shape)

    def generate_playlist_recos(df, features, nonplaylist_features):
        """ 
        Pull songs from a specific playlist.

        Parameters: 
            df (pandas dataframe): spotify dataframe
            features (pandas series): summarized playlist feature
            nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
            
        Returns: 
            non_playlist_df_top_40: Top 40 recommendations for that playlist
        """
        
        non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
        nonplaylist_features=nonplaylist_features[nonplaylist_features['id'].isin(df['id'].values)]
        print(len(non_playlist_df))
        print(len(nonplaylist_features.drop('id', axis = 1).values))
        non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values  , features.values.reshape(1, -1))[:,0]
        non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
        non_playlist_df_top_40['images'] = non_playlist_df_top_40['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
        non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(lambda x: sp.track(x)['external_urls']['spotify'])
        
        return non_playlist_df_top_40

    work_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_WORKOUT, complete_feature_set_nonplaylist_WORKOUT)
    return work_top40

work_top40=generate()

recommendation_list=[]

for i in range(0,40):
    recommendation_list.append({'name':str(list(work_top40['name'])[i]),
    'images':str(list(work_top40['images'])[i]),
    'artists':str(list(work_top40['artists'])[i].split("'")[1]),
    'url':str(list(work_top40['url'])[i]),
    })

print(recommendation_list)

print(work_top40['name'])