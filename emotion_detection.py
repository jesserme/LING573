#!/usr/bin/python3

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import nltk.tokenize

import pickle
import os.path
import numpy

import conf

def load_data(input_filename):
    '''
    Read the TSV file `input_filename' and return a map of
    the form:
    'tweet_id': tuple([event, tweet, offensive, emotion])
    '''

    tweets = {}

    with open(input_filename, "r", encoding="utf-8") as input_file:

        seen_header = False


        for line in input_file:

            #skip header
            if (not seen_header ):
                seen_header = True
                continue

            line = line.strip()

            tweet_id, event, tweet, offensive, emotion = line.split("\t")

            tweets[tweet_id] = tuple([event, tweet, offensive, emotion])

    return tweets


def train_naive_bayes_1(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    train a naive bayes model on the 'event' and 'offensive' features.
    Return a tuple with 3 values:
    1.) the trained model
    2.) a lookup table for event_string => numeric_code
    3.) a lookup table for offensive_string => numeric_code
    '''

    X_feats = []
    y_pred  = []

    event_to_code = {}
    offensive_to_code = {}
    #emotion_to_code = {}

    #convert event and offensive features
    #to numeric codes
    for tweet_id in sorted(train_dataset):

        event = train_dataset[tweet_id][0].lower()
        offensive = train_dataset[tweet_id][2].lower()

        if ( event not in event_to_code ):
            event_to_code[event] = len(event_to_code)

        if ( offensive not in offensive_to_code ):
            offensive_to_code[offensive] = len(offensive_to_code)

    for tweet_id in train_dataset:

        event_code = event_to_code[train_dataset[tweet_id][0].lower()]
        offensive_code = offensive_to_code[train_dataset[tweet_id][2].lower()]

        X_feats.append([event_code, offensive_code])
        y_pred.append(train_dataset[tweet_id][3].lower())

    model = MultinomialNB()
    model.fit(X_feats, y_pred)

    return tuple([model,event_to_code, offensive_to_code])

def get_naive_baiyes_1(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    check if a Naive Bayes model has already been trained on the
    'event' and 'offensive' features.
    If yes, return the trained model
    If not, train and save the model as a pickle in 'naive_bayes_1.pickle'.
   
    Return a tuple with 3 values:
    1.) the trained model
    2.) a lookup table for event_string => numeric_code
    3.) a lookup table for offensive_string => numeric_code
    '''

    model = None

    if ( os.path.exists("naive_bayes_1.pickle") ):

        with open("naive_bayes_1.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_naive_bayes_1(train_dataset)

        with open("naive_bayes_1.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model


def test_naive_bayes_1(test_dataset, naive_bayes_model):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    and a model+params 'naive_bayes_model` which is trained on the
    'event' and 'offensive' features, return the accuracy of the model.
    '''

    model, event_to_code, offensive_to_code = naive_bayes_model

    X_feats = []
    y_pred = []

    for tweet_id in test_dataset:

        event = test_dataset[tweet_id][0].lower()
        offensive = test_dataset[tweet_id][2].lower()

     
        event_code = -1

        if ( event in event_to_code ):
            event_code = event_to_code[event]

        offensive_code = -1

        if ( offensive in offensive_to_code ):
            offensive_code = offensive_to_code[offensive]

        X_feats.append([event_code, offensive_code])
        y_pred.append(test_dataset[tweet_id][3].lower())
        
    return model.score(X_feats, y_pred)

def get_tweet_feats(train_dataset):
    '''
    Returns the bag of words features for the tweets in 'train_dataset`s
    Does pre-processing and stop words removal.
    '''
 
    unique_words = set()
    train_data = {}

    word_freqs = {}

    for tweet_id in train_dataset:

        tweet = train_dataset[tweet_id][1].lower()
        emotion = train_dataset[tweet_id][3].lower()

        words = nltk.tokenize.word_tokenize(tweet)
        cleaned_words = [word for word in words if word not in conf.stop_words]

        train_data[tweet_id] = tuple([cleaned_words, emotion])

        num_cleaned_words = len(cleaned_words)

        for word_i in range(0, num_cleaned_words):

            word = cleaned_words[word_i]
            unique_words.add(word)

            if ( word not in word_freqs ):
                word_freqs[word] = 0

            word_freqs[word] += 1

    #remove rare words/phrases (< 5% occurence)
    min_threshold = len(train_dataset) * 0.05

    for word in word_freqs:

        if ( word_freqs[word] <= min_threshold ):

            unique_words.remove(word)

    X_feats = []
    y_pred  = []

    for tweet_id in train_data:

        feats = []

        for word in sorted(unique_words):

            value = 0
            if ( word in train_data[tweet_id][0] ):
                value = 1

            feats.append(value)

        X_feats.append(feats)
        y_pred.append(train_data[tweet_id][1])

    return tuple([X_feats, y_pred, unique_words])

def train_naive_bayes_2(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    train a Naive bayes model on the tweet feature.
    Return a tuple containing the following values:
    1.) the trained model
    2.) a set with unique words seen
    '''
    
    X_feats, y_pred, unique_words = get_tweet_feats(train_dataset)

    model = BernoulliNB()
    model.fit(X_feats, y_pred)

    return tuple([model, unique_words])

def get_naive_baiyes_2(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    check if a Naive Bayes model has already been trained on the
    'tweet' feature
    If yes, return the trained model+param
    If not, train and save the model as a pickle in 'naive_bayes_2.pickle'.
   
    Return a tuple with 2 values:
    1.) the trained model
    2.) a set with the unique words(dictionary) for the model
    '''
    model = None

    if ( os.path.exists("naive_bayes_2.pickle") ):

        with open("naive_bayes_2.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_naive_bayes_2(train_dataset)

        with open("naive_bayes_2.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model

def test_BoW_model(test_dataset, model):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    and a model+params 'model`,
    return the accuracy of the provided model
    '''
    model, unique_words = model

    test_data = {}

    for tweet_id in test_dataset:

        tweet = test_dataset[tweet_id][1].lower()
        emotion = test_dataset[tweet_id][3].lower()

        words = nltk.tokenize.word_tokenize(tweet)
        cleaned_words = [word for word in words if word not in conf.stop_words]

        all_ngrams = []

        all_ngrams.extend(cleaned_words)
        
        test_data[tweet_id] = tuple([cleaned_words, emotion])

    X_feats = []
    y_pred  = []

    for tweet_id in test_data:

        feats = []

        for word in sorted(unique_words):

            value = 0
            if ( word in test_data[tweet_id][0] ):
                value = 1

            feats.append(value)

        X_feats.append(feats)
        y_pred.append(test_data[tweet_id][1])


    return model.score(X_feats, y_pred)   


def test_naive_bayes_3(test_dataset, model_1, model_2):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    and 2 Naive Bayes models + their params ('model_1` and 'model_2`),
    test the performance of an ensemble of the 2 Naive Bayes models
    with the 2 models equally weighted.
    Return the accuracy of the ensemble.
    '''
    event_offensive_model, event_to_code, offensive_to_code = model_1
    tweet_model, unique_words = model_2

    #X_feats = []
    #y_pred = []

    total_correct = 0

    for tweet_id in test_dataset:

        event = test_dataset[tweet_id][0].lower()
        offensive = test_dataset[tweet_id][2].lower()
        emotion = test_dataset[tweet_id][3].lower()

        event_code = -1

        if ( event in event_to_code ):
            event_code = event_to_code[event]

        offensive_code = -1

        if ( offensive in offensive_to_code ):
            offensive_code = offensive_to_code[offensive]

        #predict with the event and offensive features model
        probabilities = {}
        pred = event_offensive_model.predict_proba(numpy.array([event_code, offensive_code]).reshape(1, -1))[0].tolist()


        classes_list = event_offensive_model.classes_.tolist()



        for emotion_i in range(0, len(classes_list)):
            #each model has an equal weight
            probabilities[classes_list[emotion_i]] = pred[emotion_i] * 0.5

        tweet = test_dataset[tweet_id][1].lower()

        words = nltk.tokenize.word_tokenize(tweet)
        cleaned_words = [word for word in words if word not in conf.stop_words]

        all_ngrams = []

        num_cleaned_words = len(cleaned_words)

        #adding bigrams and trigrams hurts accuracy
        '''
        #bigrams

        for word_i in range(0, num_cleaned_words-1):

            all_ngrams.append("{0} {1}".format(cleaned_words[word_i], cleaned_words[word_i + 1]))

        #trigrams
        for word_i in range(0, num_cleaned_words-2):

            all_ngrams.append("{0} {1} {2}".format(cleaned_words[word_i], cleaned_words[word_i + 1], cleaned_words[word_i + 2]))
        '''

        all_ngrams.extend(cleaned_words)

        feats = []

        for word in sorted(unique_words):

            value = 0

            if ( word in all_ngrams ):
                value = 1

            feats.append(value)

        #predict with the tweet feature model
        pred = list(tweet_model.predict_proba(numpy.array(feats).reshape(1, -1))[0])

        classes_list = list(tweet_model.classes_)

        for emotion_i in range(0, len(classes_list)):
            #each model has an equal weight
            probabilities[classes_list[emotion_i]] += pred[emotion_i] * 0.5

        best_fit = max(probabilities, key=lambda x: probabilities[x])

        if ( best_fit == emotion ):
            total_correct += 1

    return total_correct / len(test_dataset)

def train_knn(train_dataset):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    train a K-Nearest Neighbours classifier
    Return a tuple with:
    1.) the kNN classifier model
    2.) the set of unique words

    '''

    X_feats, y_pred, unique_words = get_tweet_feats(train_dataset)

    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(X_feats, y_pred)

    return tuple([model, unique_words])


def get_knn(train_dataset):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    check if the KNN model has already been trained.
    If yes, return the model
    If not, train the model and pickle it to 'knn.pickle'
    Return the KNN model
    '''
    model = None

    if ( os.path.exists("knn.pickle") ):

        with open("knn.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_knn(train_dataset)

        with open("knn.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model


dev_set = load_data("dev.tsv")
train_set = load_data("train.tsv")

#event and offensive feature
model_and_params_1 = get_naive_baiyes_1(train_set)
print("Event & offensive features nodel accuracy:", test_naive_bayes_1(dev_set, model_and_params_1))


#tweet features
model_and_params_2 = get_naive_baiyes_2(train_set)
print("Tweet feature model acccuracy:", test_BoW_model(dev_set, model_and_params_2))

#event/offensive & tweet ensemble
print("Event/offensive & tweet ensemble accuracy:", test_naive_bayes_3(dev_set, model_and_params_1, model_and_params_2))

#knn model
knn_model_and_params = get_knn(train_set)
print("KNN model:",  test_BoW_model(dev_set, knn_model_and_params))
