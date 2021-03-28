
import nltk
from os import getcwd


# if you dont have this files in your local machine uncomment 2 below lines 
nltk.download('twitter_samples')
nltk.download('stopwords')


# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
# this enables importing of these files without downloading it again when we refresh our workspace
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
from utils import process_tweet, build_freqs


# Prepare the data
# The `twitter_samples` contains subsets of 5,000 positive tweets, 5,000 negative tweets, and the full set of 10,000 tweets.  
# If you used all three datasets, we would introduce duplicates of the positive tweets and negative tweets.  
# You will select just the five thousand positive tweets and five thousand negative tweets.

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')



# split the data into two pieces, one for training and one for testing (validation set) 
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg


# combine positive and negative labels 
# we know that first part of train and test is positive 
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)




#  Create the frequency dictionary using the imported `build_freqs()` function.  
#  The `freqs` dictionary is the frequency dictionary that's being built. 
#  The key is the tuple (word, label), such as ("happy",1) or ("happy",0).  The value stored for each key is the count of how many times the word "happy" was associated with a positive label, or how many times "happy" was associated with a negative label.

# create frequency dictionary
freqs = build_freqs(train_x, train_y)


# Process tweet
# The given function `process_tweet()` tokenizes the tweet into individual words, removes stop words and applies stemming.


# The sigmoid function is defined as: 
# h(z) = 1/(1+exp^(-z)) 
# It maps the input 'z' to a value that ranges between 0 and 1, and so it can be treated as a probability. 
# You will want this function to work if z is a scalar as well as if it is an array.

def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    h = 1/(1+ np.exp(-z))    
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    '''
    # get 'm', the number of rows in matrix x
    m,_ = x.shape 

    for i in range(num_iters):
        
        # get z, the dot product of x and theta
        z = x @ theta
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = (-1/m) * (((y.transpose())@(np.log(h))) + (((1-y).transpose())@ (np.log(1-h))))

        # update the weights theta
        theta = theta - ((alpha/m) * ((x.transpose()) @(h-y)))

        # if you want fallow the decresing cost uncomment bellow line
        # print(J)

    J = float(J)
    return J, theta




 
# Given a list of tweets, extract the features and store them in a matrix. You will extract two features.
#  The first feature is the number of positive words in a tweet.
#  The second feature is the number of negative words in a tweet. 
# Then train your logistic regression classifier on these features.
#  Test the classifier on a validation set. 



# This function takes in a single tweet.
# Process the tweet using the imported `process_tweet()` function and save the list of tweet words.
# Loop through each word in the list of processed words
# For each word, check the `freqs` dictionary for the count when that word has a positive '1' label. (Check for the key (word, 1.0)
# the same for the count for when the word is associated with the negative label '0'. (Check for the key (word, 0.0).) 


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        if (word,1.0) in freqs.keys():
            x[0,1] += freqs[(word,1.0)]
        
        # increment the word count for the negative label 0
        if (word,0.0) in freqs.keys():
            x[0,2] += freqs[(word,0.0)]
        
    # return features as a row by 3 columns 
    assert(x.shape == (1, 3))
    return x



# To train the model:
#  Stack the features for all training examples into a matrix `X`. 
#  Call `gradientDescent`, which I've implemented above.


# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print("\n\n")
print("".center(50,'*'))
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
print("".center(50,'*'))
print("\n\n")


# Predict whether a tweet is positive or negative.
#  Given a tweet, process it, then extract the features.
#  Apply the model's learned weights on the features to get the logits.
#  Apply the sigmoid to the logits to get the prediction (a value between 0 and 1).

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(x @ theta)

    return y_pred




# Check performance using the test set
# After training your model using the training set above, check how your model might perform on real, unseen data, by testing it against the test set.
#  Given the test data and the weights of your trained model, calculate the accuracy of your logistic regression model. 
#  Use your `predict_tweet()` function to make predictions on each tweet in the test set.
#  If the prediction is > 0.5, set the model's classification `y_hat` to 1, otherwise set the model's classification `y_hat` to 0.
#  A prediction is accurate when `y_hat` equals `test_y`.  Sum up all the instances when they are equal and divide by `m`.

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    #  y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    count = 0
    y_real = list(test_y)
    for i in range(len(y_hat)):
        if y_real[i] == y_hat[i]:
            count +=1
        
    accuracy = count/ len(y_hat)
    return accuracy


print("\n\n")
print("".center(50,'*'))
tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

print("".center(50,'*'))
print("\n\n")



# Feel free to change the tweet below, its really Fun
my_tweet = input("enter you message to chek it if its positive or negetive: \n")
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)

if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')

