# NLP with Classification and Vector Spaces

## Translation as linear transformation of embeddings
In this project, I use linear transformation of word embeddings to translate English words to French.<br>

* Generate embedding matrices from a python dictionary 
* Compute the gradient of loss function(square Frobenius) norm in respect to transform matrix R
* Find the optimal R with gradient descent with a fixed number of iterations 
* Search for the translation embedding by K-NN(1-NN)
* Test our translation and compute its accuracy.

## Finding the most similar tweets with LSH(locality sensitive hashing)
In this project, I use LSH instead of normal KNN to find the most similar tweet to a specific tweet.
* Create document(tweet) embeddings
* Calculate hash number for a vector
* Implement hash buckets
* Create hash tables


## Logistic Regression Classifier-Sentiment Analysis
in this project, I use logistic Regression for sentiment analysis on tweets(nltk: twitter_samples). Given a tweet,  will decided if it has a positive sentiment or a negative one.

* Process tweets: eliminate handles and URLs, tokenize the tweet into individual words, remove stop words and apply stemming.
* Extract features for logistic regression
* Implement logistic regression from scratch
* Apply logistic regression on a natural language processing task
* Test logistic regression and calculate the accuracy 
* Predict on your own tweet


## Naive Bayes Classifier-Sentiment Analysis
in this project, I use Naive Bayes for sentiment analysis on tweets(nltk: twitter_samples). Given a tweet,  will decided if it has a positive sentiment or a negative one. 

* Process tweets: eliminate handles and URLs, tokenize the tweet into individual words, remove stop words and apply stemming.
* Calculate logPrior and loglikelihood for each word
* Train a Naive Bayes model on a sentiment analysis task
* Test Naive Bayes and and calculate the accuracy 
* Predict on your own tweet

