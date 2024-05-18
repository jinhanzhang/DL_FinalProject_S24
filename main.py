from ast import arg
from math import e
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain
from ktrain import text
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import random
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math


def parse_option():
    parser = argparse.ArgumentParser(description='FoV')
    # basic config
    parser.add_argument('--model', type=str, default='fasttext',
                        help='model name, options: [fasttext,nbsvm,logreg,bigru,bert,distilbert]')
    parser.add_argument('--root_path', type=str, default=f'{os.getcwd()}', help='root path of the proj')
    parser.add_argument('--data_path', type=str, default='/scratch/jz5952/DL/movies_review.csv', help='data file')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--genre', type=bool, default=False, help='include genre of the movie')
    # parser.add_argument('--metadata', type=str, default=2, help='history data time')
    return parser.parse_args()
    
class MovieReviewDataset(Dataset):
    def __init__(self, dataframe, device):
        self.data = dataframe
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        # sample = {
        #     'movie_title': row['movie_title'],
        #     'genre': row['genre'],
        #     'date': row['date'],
        #     'username': row['username'],
        #     'review': row['review'],
        #     'helpful': float(row['helpful']),
        #     'total': float(row['total']),
        #     'rating': float(row['rating'])
        # }
        review = row['review']
        rating = torch.zeros(1,10,dtype=int)
        rating[:,int(row['rating'])-1] = 1
        return review, rating
    
    
def evaluate(test_df_list, predictor,train_std):
    
    correct = 0
    total = 0
    losses = []
    all_ratings = []
    all_predictions = []
    for idx, test_df in enumerate(test_df_list):
        print("movie title: ", )
        ratings = []
        predictions = []
        loss = 0
        for i in range(len(test_df)):
            data = test_df.iloc[i].values
            review = data[0]
            rating = data[1:].astype(int).argmax()
            prediction = predictor.predict(review)
            # print("rating!!!!!!!!!!!!: ",rating)
            # print("prediction!!!!!!!!!!!!: ",prediction)
            ratings.append(rating)
            predictions.append(int(prediction))
            if prediction == rating:
                correct += 1
            total += 1
        predictions = np.array(predictions)
        ratings = np.array(ratings)
        loss = math.sqrt(((predictions - ratings)**2).mean())/train_std
        losses.append(loss)
        all_ratings.append(ratings.mean())
        all_predictions.append(predictions.mean())
    return losses, all_ratings, all_predictions
    
    
if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    id = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    args = parse_option()
    saved_path = f'saved_results/{args.model}_{args.batch_size}'
    if args.genre is True:
        saved_path = f'saved_results/{args.model}_{args.batch_size}_genre'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PROJECT_PATH = args.root_path
    BATCH_SIZE = args.batch_size
    TEST_NUM = 10
    preprocess_mode = 'standard'
    ngram_range=3
    max_len = 1000
    if args.model == 'bert':
        preprocess_mode='bert'
        max_len = 512
    elif args.model == 'distilbert':
        preprocess_mode='distilbert'
    if args.model == 'bigru':
        ngram_range=1
    # dataset

    movies = pd.read_csv('/scratch/jz5952/DL/movies_review.csv')
    movies.replace('Null',np.nan,inplace=True)
    movies=movies.dropna()
    movies['title']=movies['title'].str.rstrip('\n')
    if args.genre is False:
        movies['review']=movies['title']+movies['Review']
    else:
        movies['review']=movies['genre']+movies['title']+movies['Review']
    movies['review'] = movies['review'].apply(lambda x: x[:max_len])
    movies.drop(columns=['title','Review','username'],inplace=True)
    movies = movies.astype({'rating': 'int64'})
    random_test_title_list = []
    for i in range(TEST_NUM):
        random_test_title = np.random.choice(movies['movie_title'].unique())
        random_test_title_list.append(random_test_title)
    # overwrite by hand
    # random_test_title_list = ['Jaws', 'A Quite Place','Hot Fuzz','Baby Driver','Bad Teacher','Inglourious Basterds','Foxcatcher', 'Face_Off','Inside Man', 'Departures', 'Empire of the Sun','Gladiator']
    # random_test_title_list = ['A Quite Place', 'Jaws','Hot Fuzz','Baby Driver','Bad Teacher','Inglourious Basterds']
    random_test_title_list = ['A Quite Place', 'Jaws']
    
    print(f"Randomly selected movie title for testing: {random_test_title_list}")
    val_train_data = movies[~movies.movie_title.isin(random_test_title_list)]
    random_val_title = np.random.choice(val_train_data['movie_title'].unique())
    print(f"Randomly selected movie title for validation: {random_val_title}")
    val_data = val_train_data[val_train_data['movie_title'] == random_val_title]
    train_data=val_train_data[val_train_data['movie_title'] != random_val_title]

    x_train = train_data['review'].values
    y_train_value = train_data['rating'].values
    train_std = y_train_value.std()
    y_train = np.zeros((y_train_value.size, y_train_value.max()),dtype=int) 
    y_train[np.arange(y_train_value.size), y_train_value-1] = 1
    x_val = val_data['review'].values
    y_val_value = val_data['rating'].values
    y_val = np.zeros((y_val_value.size, y_val_value.max()),dtype=int)
    y_val[np.arange(y_val_value.size), y_val_value-1] = 1
    
    
    # create df for df import method
    labels = ['1','2','3','4','5','6','7','8','9','10']
    train_df = pd.DataFrame({'text':x_train,'1':y_train[:,0],'2':y_train[:,1],'3':y_train[:,2],'4':y_train[:,3],'5':y_train[:,4],'6':y_train[:,5],'7':y_train[:,6],'8':y_train[:,7],'9':y_train[:,8],'10':y_train[:,9]})
    print(train_df.head())
    val_df = pd.DataFrame({'text':x_val,'1':y_val[:,0],'2':y_val[:,1],'3':y_val[:,2],'4':y_val[:,3],'5':y_val[:,4],'6':y_val[:,5],'7':y_val[:,6],'8':y_val[:,7],'9':y_val[:,8],'10':y_val[:,9]})
    test_df_list = []
    for random_test_title in random_test_title_list:
        test_data = movies[movies['movie_title'] == random_test_title]
        x_test = test_data['review'].values
        y_test_value = test_data['rating'].values
        y_test = np.zeros((y_test_value.size, y_test_value.max()),dtype=int)
        y_test[np.arange(y_test_value.size), y_test_value-1] = 1
        test_df = pd.DataFrame({'text':x_test,'1':y_test[:,0],'2':y_test[:,1],'3':y_test[:,2],'4':y_test[:,3],'5':y_test[:,4],'6':y_test[:,5],'7':y_test[:,6],'8':y_test[:,7],'9':y_test[:,8],'10':y_test[:,9]})
        test_df_list.append(test_df)
    # trn, val, preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
    #                                       x_test=x_test, y_test=y_test,
    #                                       class_columns=['1','2','3','4','5','6','7','8','9','10'],
    #                                       preprocess_mode='standard',
    #                                       maxlen=1000)
    
    trn, val, preproc = text.texts_from_df(train_df,
                                            'text', # name of column containing review text
                                            label_columns=labels,
                                            val_df=val_df,
                                            maxlen=max_len, 
                                            max_features=100000,
                                            preprocess_mode=preprocess_mode,
                                            ngram_range=ngram_range)
    
    text.print_text_classifiers()
    # load the model
    model = text.text_classifier(args.model, trn, preproc=preproc,metrics=['mse'])
    learner = ktrain.get_learner(model, train_data=trn, val_data=val)
    learner.lr_find(show_plot=True)
    lr_plot = learner.lr_plot(return_fig=True)
    lr_plot.show()
    lr_plot.savefig(f'{saved_path}/lr.png')
    learner.fit(0.001, 3, cycle_len=1, cycle_mult=2)
    learner.view_top_losses(n=1, preproc=preproc)
    loss_plot = learner.plot(plot_type='mse', return_fig=True)
    loss_plot.show()
    loss_plot.savefig(f'{saved_path}/losses.png')
    p = ktrain.get_predictor(learner.model, preproc)
    # prediction = p.predict(x_test[0])
    # print(prediction)
    # print(y_test[0])
    loss, ratings, predictions = evaluate(test_df_list, p,train_std)
    print("model name:", args.model)
    print("genre: ", args.genre)
    print("movie title: ",random_test_title_list)
    print("predict overall score:" ,predictions)
    print("actual overall score:" ,ratings)
    print(f"Loss: {loss}")