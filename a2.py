import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import LinearSVC

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    
    inputdata = inputfile.readlines()
    
    # I clean and lowercase the data and create a structure where each row is an embedded list.
    clean_data = []
    for s in inputdata:
        short_s = s.replace('\n', '')
        clean_row = short_s.split("\t")
        lower = clean_row[2].lower()
        clean_row[2] = lower
        clean_data.append(clean_row)
    
    # In order to lemmatize correctly, I provide WordNet's POS-tags to the lemmatizer (instead of Penn Treebank) for each word.
    lemmatizer = WordNetLemmatizer()
    
    for row in clean_data[1:]:
        if row[3].startswith('J'):
            l = lemmatizer.lemmatize(row[2],wordnet.ADJ)
            row[2] = l
        elif row[3].startswith('V'): 
            l = lemmatizer.lemmatize(row[2],wordnet.VERB)
            row[2] = l
        elif row[3].startswith('N'): 
            l = lemmatizer.lemmatize(row[2],wordnet.NOUN)
            row[2] = l
        elif row[3].startswith('R'): 
            l = lemmatizer.lemmatize(row[2],wordnet.ADV)
            row[2] = l
        else:
            l = lemmatizer.lemmatize(row[2])
            row[2] = l

    return clean_data[1:]

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []
    
    for index, row in enumerate(data):
        if row[4].startswith('B'):
            neclass = row[4][2:]
            features = []
            sentence = row[1]
            
            # I append the previous 5 words to features, if they belong to the same sentence. 
            # If not, I append corresponding number of start-symbols.
            for n in reversed(range(1,6)):
                if data[index-n][1] == sentence:
                    features.append(data[index-n][2])
                else:
                    features.append('<S>')
                    
            # I find the end of the named entity.
            counter = 1
            while data[index+counter][4].startswith('I'):
                counter = counter +1
            end = (index+counter)-1
            
            # I append the 5 words after the end of the named entity to features, if they belong to the same sentence.
            # If not, I append corresponding number of end-symbols.
            if end+5 < len(data):
                for n in range(1,6):
                    if data[end+n][1] == sentence:
                        features.append(data[end+n][2])
                    else:
                        features.append('</S>')
                        
            # This is a special case for the last named entity to avoid an index error.
            else:
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
            
            instances.append(Instance(neclass, features))
            
    return instances

# Code for part 3
def create_table(instances):
    
    # I create a dictionary of the 3000 most frequent words in the features.
    classes = []
    wordSet = set()
    for instance in instances:
        wordSet = set(instance.features).union(wordSet)
        classes.append(instance.neclass)
    
    wordCount = dict.fromkeys(wordSet, 0)
    
    for instance in instances:
        for word in instance.features:
            wordCount[word] += 1
    
    most_freq = sorted(wordCount, key=wordCount.get, reverse=True)[:3000]
    
    # I create a dictionary of the word counts for each feature and append them to a list, in order to create a DataFrame.
    featureDicts = []
    for instance in instances:
        d = dict.fromkeys(most_freq, 0)
        for word in instance.features:
            if word in most_freq:
                d[word] += 1
        featureDicts.append(d)
    
    df = pd.DataFrame(featureDicts)
    df.insert(0, "class", classes, True)

    return df

def ttsplit(bigdf):
    
    # I split the big data frame randomly into 80% train data and 20% test data.
    df_train=bigdf.sample(frac=0.8)
    df_test=bigdf.drop(df_train.index)
    
    # I reset the indices of the new data frames.
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
        
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
def confusion_matrix(truth, predictions):
    
    # I use Sklearn's confusion_matrix and add the names of the classes.
    classes = ['art', 'eve', 'geo', 'gpe', 'nat', 'org', 'per', 'tim']
    df = pd.DataFrame(cm(truth, predictions, labels=classes))
    df.index = classes
    df.columns = classes
    return df

# Code for bonus part B

def bonus_create_instances(data):
    instances = []
    
    for index, row in enumerate(data):
        if row[4].startswith('B'):
            neclass = row[4][2:]
            features = []
            sentence = row[1]
        
            for n in reversed(range(1,6)):
                if data[index-n][1] == sentence:
                    features.append(data[index-n][2])
                    features.append(data[index-n][3])# I add this line to add the POS-tag.
                else:
                    features.append('<S>')
            
            counter = 1
            while data[index+counter][4].startswith('I'):
                counter = counter +1
            end = (index+counter)-1
            
            if end+5 < len(data):
                for n in range(1,6):
                    if data[end+n][1] == sentence:
                        features.append(data[end+n][2])
                        features.append(data[end+n][3])# I add this line to add the POS-tag.
                    else:
                        features.append('</S>')
            else:
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
                features.append('</S>')
            
            instances.append(Instance(neclass, features))
            
    return instances

def bonusb(filename):
    gmbfile = open('/scratch/lt2222-v21-resources/GMB_dataset.txt', "r")
    inputdata = preprocess(gmbfile)
    gmbfile.close()
    
    # I use the bonus version of create_instances to include both words and POS-tags.
    instances = bonus_create_instances(inputdata)
    print('instances[20:30] BONUS')
    print()
    print(instances[20:30])
    print()
    
    bigdf = create_table(instances)
    
    train_X, train_y, test_X, test_y = ttsplit(bigdf)
    
    model = LinearSVC()
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    cm_test = confusion_matrix(test_y, test_predictions)
    cm_train = confusion_matrix(train_y, train_predictions)
    
    print('Confusion matrix test data BONUS')
    print()
    print(cm_test)
    print()
    print('Confusion matrix train data BONUS')
    print()
    print(cm_train)
    
    return
