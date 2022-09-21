from collections import UserList
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm
from typing import *
import time

class DataLoader():
    def __init__(self, 
                 config : Dict, 
                 make_dataset : bool = True
                 ) -> (None):
        self.config = config
        
        self.batch_size = config["batch_size"]
        self.split_ratio = config["split_ratio"]
        
        self.movies_path = config["movies_path"]
        self.ratings_path = config["ratings_path"]
        self.users_path = config["users_path"]
        
        self.negative_sample_count = config["negative_sample_count"]
        
        self.negative_sample = {}        
        
        self._load_()
        self._init_data_()
        
        if make_dataset:
            self._make_dataset_()
    
    def _load_(self):
        
        movie_columns = ['MovieID', 'Title', 'Genres']
        movie_column_dtypes = [str, str, str]
        self.movies_data = pd.read_csv(self.movies_path, encoding='latin-1', sep='::',
                                       dtype={k: v for k, v in zip(movie_columns, movie_column_dtypes)}, names=movie_columns,  engine='python')
        
        rating_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        rating_column_dtypes = [str, str, int, int]
        ratings = pd.read_csv(self.ratings_path, encoding='latin-1', sep='::',
                                        dtype={k: v for k, v in zip(rating_columns, rating_column_dtypes)}, names=rating_columns, engine='python')
        ratings = ratings.sort_values(by='Timestamp', ascending=True)
        self.ratings_data = ratings
        
        user_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        user_column_dtypes = [str, str, str, str, str]
        self.users_data = pd.read_csv(self.users_path, encoding='latin-1', sep='::',
                                      dtype={k: v for k, v in zip(user_columns, user_column_dtypes)}, names=user_columns, engine='python')
    
    def _init_data_(self):
        self.movie_ids = self.movies_data['MovieID'].unique()
        self.user_ids = self.users_data['UserID'].unique()
        self.movie_len = len(self.movie_ids)
    
    def _make_dataset_(self):
        self.sessions = self.ratings_data[['UserID', 'MovieID']].groupby('UserID').aggregate(list)
        train_user, valid_user = self.split_user()
        
        self.train_x, self.train_y = self.slice_sequence(train_user)
        self.valid_x, self.valid_y = self.slice_sequence(valid_user)
        
        
        
    def split_user(self):
        user = np.array(self.users_data['UserID'].unique())
        
        np.random.seed(self.config["numpy_seed"])
        np.random.shuffle(user)
        
        total_len = len(user)
        
        train_user = user[:int(total_len * self.split_ratio)]
        valid_user = user[int(total_len * self.split_ratio):]
        
        return train_user, valid_user
    
    def get_movie_by_userid(self, user_id):
        user_ratings = self.ratings_data[self.ratings_data['UserID'] == user_id]    ## collect movie list and ratings for the user id
        user_positive_movies = user_ratings['MovieID']
        
        return user_positive_movies
    
    
    def slice_sequence(self, user_list):
        sample_ratio = 10
        sequence_length = self.config["sequence_length"]
        # sequence_length = 5

        input_x = []
        input_y = []

        session_partial = self.sessions.copy()['MovieID'][user_list]

        for key in tqdm(session_partial.keys()):
            movie_list = session_partial[key]
            movie_length = len(movie_list)
            
            if movie_length >= 2:
                sample_count = (movie_length) // sample_ratio
                if sample_count == 0:
                    sample_count = 1

            last_pivots = random.sample(range(2, movie_length), sample_count)
            
            for last_idx in last_pivots:
                start_idx = last_idx - sequence_length + 1
                if start_idx < 0: 
                    sequence = [* (["pad"] * (-start_idx)), *movie_list[0 : last_idx]]
                else:
                    sequence = movie_list[start_idx : last_idx]
                label = movie_list[last_idx]

                input_x.append(sequence)
                input_y.append(label)

        return input_x, input_y
            
            
    # def add_session_in_graph(self, session):

    #     sequence_idx = self.string_lookup.str_to_idx(session)

    #     sequence_idx = sequence_idx.numpy()
    #     for i in range(len(sequence_idx)-1):
    #         self.graph[sequence_idx[i]][sequence_idx[i+1]] += 1


    '''
    APIs
    '''
    #############################################
    def get_dataset(self, phase, batch_size = None) -> (tf.data.Dataset):
        '''
        Get dataset from dataloader
        Args : 
            phase : 'train' or 'valid'
            batch_size : Int value
        '''
        
        if batch_size == None:
            batch_size = self.batch_size
        
        if phase == "train":
            # x = self.string_lookup.str_to_idx(self.train_x)
            # y = self.string_lookup.str_to_idx(self.train_y)
            
            x = self.train_x
            y = self.train_y
            
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset\
                        .batch(batch_size, drop_remainder=True)\
                        .shuffle(buffer_size = len(x))\
                        .cache()\
                        .prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        elif phase == "valid":
            # x = self.string_lookup.str_to_idx(self.valid_x)
            # y = self.string_lookup.str_to_idx(self.valid_y)

            x = self.valid_x
            y = self.valid_y
            
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset\
                        .batch(batch_size, drop_remainder=True)\
                        .cache()\
                        .prefetch(tf.data.AUTOTUNE)
            
            return dataset

    def get_history(self
                    ) -> (pd.DataFrame):
        '''
        Get all user history from dataloader
        '''
        
        return self.sessions.copy()['MovieID']

    def get_items(self
                  ) -> (List):
        return self.movie_ids
    
    def get_item_len(self):
        return self.movie_len
    ###################################
    
    def get_movie(self, id):
        idx = self.movies_data.index[self.movies_data[0] == id]
        idx = idx.tolist()[0]
        movie = self.movies_data.iat[idx, 1]
        genre = self.movies_data.iat[idx, 2]
        
        return movie, genre


    def get_movie_ids(self):        
        return self.movie_ids
    
    def get_graph(self):
        return self.graph
    
    def get_user_movie(self, user_id):
        ratings = self.ratings_data.iloc[:, 0:3]
                
        user_ratings = ratings[ratings[0] == user_id]   ## collect movie list and ratings for the user id
        
        user_positive_movies = np.array(user_ratings[1])
        user_ratings = np.array(user_ratings[2])
        
        return user_positive_movies, user_ratings


    def get_movie_length(self):
        return self.movie_length
    
    def get_user_length(self):
        return self.user_length