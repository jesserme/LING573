#!/usr/bin/python3

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from collections import defaultdict
import nltk.tokenize

import pickle
import os.path
import numpy
import sys
import os.path

import conf
import random

import csv
import re

def load_data(input_filename):
    '''
    Read the CSV file `input_filename' and return a map of
    the form:
    'tweet_id': tuple([event, tweet, offensive, emotion])
    'offensive' and 'event' are always blank("")
    (This is legacy from the earlier iter)
    '''

    tweets = {}

    with open(input_filename, "r", encoding="utf-8") as input_file:

        csv_reader = csv.reader(input_file)

        seen_header = False

        for row in csv_reader:

            #skip header
            if ( not seen_header ):

                seen_header = True
                continue

            tweet_id, emotion, tweet = row

            tweets[tweet_id] = tuple(["", tweet, "", emotion])

    return tweets


def get_other_feats(train_dataset):
    '''
    Get event, offensive, tweet length
    features.
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

        #len codes: < 120: 0; 120-180: 1; > 180: 2
        len_code = 1
        tweet_len = len(train_dataset[tweet_id][1])

        if ( tweet_len < 120 ):
            len_code = 0

        elif ( tweet_len > 180 ):
            len_code = 2

        X_feats.append([event_code, offensive_code, len_code])
        y_pred.append(train_dataset[tweet_id][3].lower())

    return tuple([X_feats, y_pred, event_to_code, offensive_to_code])

def train_svm_1(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    train an SVM Model on the 'event', 'offensive', length and
    features.
    Return a tuple with 3 values:
    1.) the trained model
    2.) a lookup table for event_string => numeric_code
    3.) a lookup table for offensive_string => numeric_code
    '''

    X_feats, y_pred, event_to_code, offensive_to_code = get_other_feats(train_dataset)


    model = SVC(kernel='rbf', gamma=0.5, probability=True )
    #model = SVC(kernel='linear', probability=True )
    model.fit(X_feats, y_pred)

    return tuple([model,event_to_code, offensive_to_code])

def get_svm_1(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    check if an SVM model has already been trained on the
    'event' and 'offensive' features.
    If yes, return the trained model
    If not, train and save the model as a pickle in 'svm_1.pickle'.
   
    Return a tuple with 3 values:
    1.) the trained model
    2.) a lookup table for event_string => numeric_code
    3.) a lookup table for offensive_string => numeric_code
    '''

    model = None

    if ( os.path.exists("svm_1.pickle") ):

        with open("svm_1.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_svm_1(train_dataset)

        with open("svm_1.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model


def test_other_feats_model(test_dataset, model):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    and a model+params 'model` which is trained on the
    'event', 'offensive', tweet length 
    features.
    Return the accuracy of the model.
    '''

    model, event_to_code, offensive_to_code = model

    X_feats = []
    y_true = []

    for tweet_id in test_dataset:

        event = test_dataset[tweet_id][0].lower()
        offensive = test_dataset[tweet_id][2].lower()

     
        event_code = -1

        if ( event in event_to_code ):
            event_code = event_to_code[event]

        offensive_code = -1

        if ( offensive in offensive_to_code ):
            offensive_code = offensive_to_code[offensive]

        #len codes: < 120: 0; 120-180: 1; > 180: 2
        len_code = 1
        tweet_len = len(test_dataset[tweet_id][1])

        if ( tweet_len < 120 ):
            len_code = 0

        elif ( tweet_len > 180 ):
            len_code = 2

        X_feats.append([event_code, offensive_code, len_code])
        y_true.append(test_dataset[tweet_id][3].lower())
    
    y_pred = model.predict(X_feats)


    print(" ".join(model.classes_.tolist()))
    print("Confusion matrix(Other Feats Model)")
    print(confusion_matrix(y_pred, y_true))

    return model.score(X_feats, y_true)

def get_tweet_feats(train_dataset):
    '''
    Returns the bag of words features for the tweets in 'train_dataset`s
    Does pre-processing and stop words removal.
    '''
 
    unique_words = set()
    train_data = {}

    all_word_freqs = {}

    for tweet_id in train_dataset:

        tweet = train_dataset[tweet_id][1].lower()

        emotion = train_dataset[tweet_id][3].lower()

        #Vietnamese stop words are more like 'stop phrases'
        tweet_dup = " " + tweet + " "

        for stop_phrase in sorted(conf.stop_words, key=lambda x: len(x), reverse=True):

            tweet_dup = re.sub("\s{0}\s".format(stop_phrase), " ", tweet_dup)

            pass

        #cleaned_words = tweet_dup.split()
        cleaned_words = tweet.split()
        num_cleaned_words = len(cleaned_words) 

        #words = nltk.tokenize.word_tokenize(tweet)
        #cleaned_words = [word for word in words if word not in conf.stop_words]

        ngrams = []

        for word_i in range(0, num_cleaned_words):

            word = cleaned_words[word_i]
            unique_words.add(word)

            if ( word not in all_word_freqs ):
                all_word_freqs[word] = 0

            all_word_freqs[word] += 1


        ngrams.extend(cleaned_words)
        train_data[tweet_id] = tuple([ngrams, emotion])

    #remove rare words/phrases (< 1% occurence)
    min_threshold = len(train_dataset) * 0.001
    #min_threshold = 0
    for word in all_word_freqs:

        if ( all_word_freqs[word] <= min_threshold ):

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

    smote_X_feats, smote_y_pred = smote(X_feats, y_pred)
    return tuple([smote_X_feats, smote_y_pred, unique_words])

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

        #Vietnamese stop words are more like 'stop phrases'
        tweet_dup = " " + tweet + " "

        for stop_phrase in sorted(conf.stop_words, key=lambda x: len(x), reverse=True):

            tweet_dup = re.sub("\s{0}\s".format(stop_phrase), " ", tweet_dup)

            pass

        #cleaned_words = tweet_dup.split()
        cleaned_words = tweet.split()
        #words = nltk.tokenize.word_tokenize(tweet)
        #cleaned_words = [word for word in words if word not in conf.stop_words]

        all_ngrams = []

        all_ngrams.extend(cleaned_words)

        test_data[tweet_id] = tuple([all_ngrams, emotion])

    X_feats = []
    y_true  = []

    for tweet_id in test_data:

        feats = []

        for word in sorted(unique_words):

            value = 0
            if ( word in test_data[tweet_id][0] ):
                value = 1

            feats.append(value)

        X_feats.append(feats)
        y_true.append(test_data[tweet_id][1])

    y_pred = model.predict(X_feats)
    print(" ".join(model.classes_.tolist()))
    print("Confusion matrix(BoW Model)")
    print(confusion_matrix(y_pred, y_true))

    return model.score(X_feats, y_true)   

def get_other_feats_2(tweet, event_to_code, offensive_to_code):
    '''
    Given a tweet, return 4 features: event, offensive, length, 
    mark use as a list.
    '''
    event = tweet[0].lower()
    offensive = tweet[2].lower()
    emotion = tweet[3].lower()

    event_code = -1

    if ( event in event_to_code ):
        event_code = event_to_code[event]

    offensive_code = -1

    if ( offensive in offensive_to_code ):
        offensive_code = offensive_to_code[offensive]

    #len codes: < 120: 0; 120-180: 1; > 180: 2
    len_code = 1
    tweet_len = len(tweet[1])

    if ( tweet_len < 120 ):
        len_code = 0

    elif ( tweet_len > 180 ):
        len_code = 2

    return [event_code, offensive_code, len_code]


def test_ensemble(test_dataset, models, model_weights):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    a list of models + their params (`models') and the corresponding
    weight of each model ('model_weights`),
    test the performance of an ensemble of the models.
    The models/weights are expected in this order:
    1.) Naive Bayes event+offensive features model
    2.) Naive bayes tweet model
    3.) kNN model 1
    4.) kNN model 2
    5.) SVM model 1
    6.) SVM model 2
    7.) SVM model 3
    Return the accuracy of the ensemble.
    '''

    #a weight for each model expected
    assert len(models) == len(model_weights), "Expected len(models) == len(model_weights) i.e a weight for each model"

    #provide at least 2 mode
    assert len(models) >= 2, "Expected at least 2 models to be provided"


    model_1 = models[0]
    event_offensive_model, event_to_code, offensive_to_code = None, None, None

    if (model_1 is not None):
        event_offensive_model, event_to_code, offensive_to_code = model_1

    model_2 = models[1]
    tweet_model, unique_words = None, None

    if ( model_2 is not None ):
        tweet_model, unique_words = model_2

    model_3 = None
    model_4 = None
    model_5 = None
    model_6 = None
    model_7 = None

    keywords = None

    if ( len(models) > 2 and models[2] is not None ):
        model_3, unique_words  = models[2]

    if ( len(models) > 3  and models[3] is not None):
        model_4, event_to_code, offensive_to_code = models[3]

    if ( len(models) > 4  and models[4] is not None):
        model_5, unique_words  = models[4]

    if ( len(models) > 5  and models[5] is not None):
        model_6, event_to_code, offensive_to_code = models[5]

    if ( len(models) > 6  and models[6] is not None):
        model_7, keywords  = models[6]

    #X_feats = []
    #y_pred = []

    total_correct = 0

    for tweet_id in test_dataset:

        emotion = test_dataset[tweet_id][3].lower()


        probabilities = defaultdict(lambda: 0)
        feats_a = []
        
        if ( event_to_code is not None ):
            feats_a = get_other_feats_2(test_dataset[tweet_id], event_to_code, offensive_to_code)

        #predict with the event and offensive features model
        if ( event_offensive_model is not None ):

            pred = event_offensive_model.predict_proba(numpy.array(feats_a).reshape(1, -1))[0].tolist()


            classes_list = event_offensive_model.classes_.tolist()

            for emotion_i in range(0, len(classes_list)):
                #each model has an equal weight
                probabilities[classes_list[emotion_i]] = pred[emotion_i] * model_weights[0]

        tweet = test_dataset[tweet_id][1].lower()

        #Vietnamese stop words are more like 'stop phrases'
        tweet_dup = " " + tweet + " "

        for stop_phrase in sorted(conf.stop_words, key=lambda x: len(x), reverse=True):

            tweet_dup = re.sub("\s{0}\s".format(stop_phrase), " ", tweet_dup)

            pass

        #cleaned_words = tweet_dup.split()
        cleaned_words = tweet.split()
        num_cleaned_words = len(cleaned_words) 

        #words = nltk.tokenize.word_tokenize(tweet)
        #cleaned_words = [word for word in words if word not in conf.stop_words]

        all_ngrams = []

        num_cleaned_words = len(cleaned_words)

        #adding bigrams and trigrams hurts accuracy
        all_ngrams.extend(cleaned_words)

        feats = []

        if ( unique_words is not None ):

            for word in sorted(unique_words):

                value = 0

                if ( word in all_ngrams ):
                    value = 1

                feats.append(value)

        feats_c = []

        if ( keywords is not None ):

            for word in sorted(keywords):

                value = 0

                if ( word in all_ngrams ):
                    value = 1

                feats_c.append(value)


        #predict with the tweet feature model
        if ( tweet_model is not None ):

            pred = list(tweet_model.predict_proba(numpy.array(feats).reshape(1, -1))[0])

            classes_list = list(tweet_model.classes_)

            for emotion_i in range(0, len(classes_list)):
                #each model has an equal weight
                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[1]


        #predict with the kNN model 1
        if ( model_3 is not None ):

            pred = list(model_3.predict_proba(numpy.array(feats).reshape(1, -1))[0])

            classes_list = list(model_3.classes_)

            for emotion_i in range(0, len(classes_list)):

                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[2]

        #predict with kNN model 2

        if ( model_4 is not None ):

            pred = model_4.predict_proba(numpy.array(feats_a).reshape(1, -1))[0].tolist()


            classes_list = model_4.classes_.tolist()

            for emotion_i in range(0, len(classes_list)):
                #each model has an equal weight
                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[3]

        #predict with SVM model 1
        if ( model_5 is not None ):

            #print("SVM ensemle check")
            pred = list(model_5.predict_proba(numpy.array(feats).reshape(1, -1))[0])

            classes_list = list(model_5.classes_)

            for emotion_i in range(0, len(classes_list)):

                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[4]

        #predict with SVM model 2
        if (model_6 is not None ):

            pred = model_6.predict_proba(numpy.array(feats_a).reshape(1, -1))[0].tolist()

            classes_list = model_6.classes_.tolist()

            for emotion_i in range(0, len(classes_list)):
                #each model has an equal weight
                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[5]

        #predict with SVM model 3
        if ( model_7 is not None ):

            #print("SVM ensemle check")
            pred = list(model_7.predict_proba(numpy.array(feats_c).reshape(1, -1))[0])

            classes_list = list(model_7.classes_)

            for emotion_i in range(0, len(classes_list)):

                probabilities[classes_list[emotion_i]] += pred[emotion_i] * model_weights[6]


        best_fit = max(probabilities, key=lambda x: probabilities[x])

        if ( best_fit == emotion ):
            total_correct += 1

    return total_correct / len(test_dataset)

def rescale(x):
    '''
    Rescale `x' to 0 | 1
    '''

    if (x > 0.5):
        return 1

    else:
        return 0

def smote(X_feats, y_preds):
    '''
    Add 'synthetic' samples to `X_feats'
    for unde-represented classes (e.g 'fear').
    Read the minimum number of samples expected
    for each class in conf.min_samples.
    For the classes that are under-represented,
    find the 5 nearest neighbours for each sample 
    in that class and create a 'synthetic' sample
    as:
    for each neighbour in 5-nearest neighbours,
        synthetic_sample = valid_sample + neighbour
        add_sample(synthetic_sample)
    '''

    new_X_feats = []
    new_y_preds = []

    sample_count = defaultdict(lambda: 0)

    for class_label in y_preds:
        sample_count[class_label] += 1

    #cache the calculated cosine similarity to
    #avoid redoing it repeatedly for the same sample pairs
    pre_computed_distances = {}

    for class_label in sample_count:

        if ( sample_count[class_label] < conf.sample_min[class_label]):
            
            num_new_samples = conf.sample_min[class_label] - sample_count[class_label] 
            new_samples_created = 0

            while ( new_samples_created < num_new_samples ):

                for sample_i in range(0, len(y_preds)):
    
                    if ( y_preds[sample_i] == class_label ):
    
                        #find 5-nearest neighbours
                        distance_all = {}
    
                        for sample_i_2 in range(0, len(y_preds)):
    
                            #don't compare the sample to itself
                            if (sample_i_2 == sample_i ):
                                continue
    
                            cos_sim = 0
                            comp_a = "{0}-{1}".format( sample_i, sample_i_2 )
                            comp_b = "{0}-{1}".format( sample_i_2, sample_i )
    
                            if ( comp_a in pre_computed_distances or comp_b in pre_computed_distances ):
                                cos_sim = pre_computed_distances[comp_a]
    
                            else:
                                #compute the cosine similarity
                                cos_sim = numpy.dot(X_feats[sample_i], X_feats[sample_i_2]) / ( numpy.linalg.norm(X_feats[sample_i]) * numpy.linalg.norm(X_feats[sample_i_2]) )
    
                                pre_computed_distances[comp_a] = cos_sim
                                pre_computed_distances[comp_b] = cos_sim
    
                            distance_all[sample_i_2] = cos_sim
    
                        closest_matches = sorted(distance_all, key=lambda x: distance_all[x], reverse=True)[0:5]
    
                        #create synthetic       
                        #neigbour_classes = []
                        sample_arr = numpy.array(X_feats[sample_i])
    
                        for close_match in closest_matches:
    
                            close_match_arr = numpy.array(X_feats[close_match])
    
                            diff = abs(sample_arr - close_match_arr)
    
                            synthetic_sample = sample_arr + (diff * random.random())
    
                            #only want 0s and 1s
                            #synthetic_sample = [rescale(x) for x in synthetic_sample]
    
                            #print(synthetic_sample)
                            new_X_feats.append(synthetic_sample)
                            new_y_preds.append(class_label)
    
                            new_samples_created += 1
    
                        if (new_samples_created >= num_new_samples ):
                            break


    new_X_feats.extend(X_feats)
    new_y_preds.extend(y_preds)

    return tuple([new_X_feats, new_y_preds])

def train_svm(train_dataset):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    train an SVM classifier
    Return a tuple with:
    1.) the SVM classifier model
    2.) the set of unique words
    '''
    X_feats, y_pred, unique_words = get_tweet_feats(train_dataset)

    model = SVC(kernel='rbf', probability=True)
    #model = SVC(kernel='linear', probability=True)
    model.fit(X_feats, y_pred)

    return tuple([model, unique_words])


def get_svm(train_dataset, force_retrain=False):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    check if the SVM model has already been trained.
    If yes, return the model
    If not, train the model and pickle it to 'svm.pickle'
    Return the SVM model and its params
    '''
    model = None

    if ( not force_retrain and os.path.exists("svm.pickle") ):

        with open("svm.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_svm(train_dataset)

        with open("svm.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model

def get_keywords_feats(train_dataset):
    '''
    Returns 200 keywords from the tweets in `train_dataset'.
    Keywords are calculated using the TF-IDF score. 
    '''

    keywords = set()

    term_freq = {}
    doc_freq = {}
    tf_idf_score = {}

    train_data = {}

    for tweet_id in train_dataset:
    
        tweet = train_dataset[tweet_id][1].lower()
        emotion = train_dataset[tweet_id][3].lower()
    
        #Vietnamese stop words are more like 'stop phrases'
        tweet_dup = " " + tweet + " "

        for stop_phrase in sorted(conf.stop_words, key=lambda x: len(x), reverse=True):

            tweet_dup = re.sub("\s{0}\s".format(stop_phrase), " ", tweet_dup)

            pass

        #cleaned_words = tweet_dup.split()
        cleaned_words = tweet.split()
        num_cleaned_words = len(cleaned_words) 

        #words = nltk.tokenize.word_tokenize(tweet)
        #cleaned_words = [word for word in words if word not in conf.stop_words]
    
        num_cleaned_words = len(cleaned_words)
    
        unique_words = set(cleaned_words)
    
        for word in unique_words:
    
            if ( word not in doc_freq ):
                doc_freq[word] = 1
            else:
                doc_freq[word] += 1
    
        for word in cleaned_words:
    
            if (word not in term_freq):
                term_freq[word] = 0
    
            term_freq[word] += 1

        train_data[tweet_id] = tuple([cleaned_words, emotion])

    for word in doc_freq:
        tf_idf_score[word] = term_freq[word] / doc_freq[word]

    keywords = set(sorted(tf_idf_score, key=lambda x:tf_idf_score[x])[0:1000])

    X_feats = []
    y_pred  = []

    for tweet_id in train_data:

        feats = []

        for word in sorted(keywords):

            value = 0

            if ( word in train_data[tweet_id][0] ):

                value = 1

            feats.append(value)

        X_feats.append(feats)
        y_pred.append(train_data[tweet_id][1])

    return tuple([X_feats, y_pred, keywords])

 
def train_svm_2(train_dataset):
    '''
    Given a training dataset 'train_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values,
    train an SVM Model on the keywords in the tweet.
    Return a tuple with 2 values:
    1.) the trained model
    2.) the keywords 
    '''

    X_feats, y_pred, keywords = get_keywords_feats(train_dataset)


    model = SVC(kernel='rbf', probability=True )
    #model = SVC(kernel='linear', probability=True )
    model.fit(X_feats, y_pred)

    return tuple([model, keywords])


def get_svm_2(train_dataset, force_retrain=False):
    '''
    Given a testing dataset 'test_dataset` which is a map of 
    'tweet_id' => tuple([event, tweet, offensive, emotion]) values
    check if the SVM model has already been trained.
    If yes, return the model
    If not, train the model and pickle it to 'svm_2.pickle'
    Return the SVM model and its params
    '''
    model = None

    if ( not force_retrain and os.path.exists("svm_2.pickle") ):

        with open("svm_2.pickle", "rb") as pickle_f:

            model = pickle.load(pickle_f)

    else:

        model = train_svm_2(train_dataset)

        with open("svm_2.pickle", "wb") as pickle_f:
            pickle.dump(model, pickle_f)

    return model


assert len(sys.argv) >= 4, "Usage: {0} <dev-set.csv> <train-set.csv> <eval-set.csv>".format(sys.argv[0])

assert os.path.exists(sys.argv[1]), "Specified dev CSV file does not exist: {0}".format(sys.argv[1])
assert os.path.exists(sys.argv[2]), "Specified train CSV file does not exist: {0}".format(sys.argv[2])
assert os.path.exists(sys.argv[3]), "Specified train CSV file does not exist: {0}".format(sys.argv[3])

dev_set = load_data(sys.argv[1])
train_set = load_data(sys.argv[2])
eval_set = load_data(sys.argv[3])

#X_feats, y_pred, unique_words = get_tweet_feats(train_set)

svm_model_and_params = get_svm(train_set, force_retrain=False)
svm_model_and_params_keywords = get_svm_2(train_set)

print("SVM model 1 (eval set):\n", test_BoW_model(eval_set, svm_model_and_params))
print("SVM model 2 (eval set):\n", test_BoW_model(eval_set, svm_model_and_params_keywords))

print("*-" * 30)

#train with train set + evalset
train_plus_eval_set = train_set
train_plus_eval_set.update(eval_set)

svm_model_and_params2 = get_svm(train_plus_eval_set, force_retrain=False)
svm_model_and_params_keywords2 = get_svm_2(train_plus_eval_set, force_retrain=False)

print("SVM model 1 (dev set):\n", test_BoW_model(dev_set, svm_model_and_params2))
print("SVM model 2 (dev set):\n", test_BoW_model(dev_set, svm_model_and_params_keywords2))

print("*-" * 30)

print("SVM 1 + SVM 2 ensemble accuracy:", test_ensemble(dev_set, [ None, None, None, None, svm_model_and_params2, None, svm_model_and_params_keywords2 ], [0.3, 0.3, 0.4, 0.4, 0.7, 0.5, 0.45]))

#
#svm model 2
#svm_model_and_params_2 = get_svm_1(train_set)
#print("SVM model 2:\n", test_other_feats_model(dev_set, svm_model_and_params_2))

#print("SVM 1 + SVM 2 ensemble accuracy:", test_ensemble(dev_set, [None, None, None, None, svm_model_and_params, svm_model_and_params_2 ], [0.3, 0.3, 0.4, 0.4, 0.5, 0.5]))

#print("SVM 1 + SVM 2 + SVM 3 ensemble accuracy:", test_ensemble(dev_set, [None, None, None, None, svm_model_and_params, svm_model_and_params_2, svm_model_and_params_3 ], [0.3, 0.3, 0.4, 0.4, 0.7, 0.5, 0.5]))

#print("SVM 1 + SVM 3:", test_ensemble(dev_set, [None, None, None, None, svm_model_and_params, None, svm_model_and_params_3 ], [0.3, 0.3, 0.4, 0.4, 0.5, 0.25, 0.75]))

#print("SVM 2 + SVM 3:", test_ensemble(dev_set, [None, None, None, None, None, svm_model_and_params_2, svm_model_and_params_3 ], [0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.4]))
