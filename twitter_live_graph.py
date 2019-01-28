#importing the libraries
import tweepy
import re
import pickle
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import tensorflow as tf
import random
import multiprocessing


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

#Initialzing the graph co-ordinates
X_graph = []
Y_graph = []

def animate(i):
    graph_data = open('namo_2019.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x,y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()
    ax1.set_title('Namo 2019')
    ax1.plot(xs,ys,c='blue')
    
def plo():
    ani = animation.FuncAnimation(fig, animate, interval = 37000)
    plt.show()


#Initializing the keys
consumer_key = 'GARJ1I8gLLTfqjQLF6j44BOjk'
consumer_secret = 'uTU7V3y9Ucy4IiQKIllyBrORyzV3ROBiPNgcCM8yTHjYccpUEL'
access_token = '736451455-pSWi5DIuqzpzHxDB5N7kgWQ6llQ0jMUeQdQLj4mf'
access_secret = 'xSoATCHnGOQbkReRaz9FdpOg9tfYmV9MLKfplZBgEhPg0'




# Initializing the API
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
query = 'modi'
api = tweepy.API(auth,timeout= 10)
    
#loading the pre trained vectorizer

with open('model.pickle','rb') as f:
    model = pickle.load(f)
    
graph = tf.get_default_graph()

#loading the pretrainedmodel
with open('vectorizer.pickle','rb') as f:
    vectorizer = pickle.load(f)

i = 0

def nlp_list():
    y_temp = 0
    while True:
        tweets = []
        for status in tweepy.Cursor(api.search,q=query + ' -filter:retweets',lang='en',result_type='recent').items(100):
            tweets.append(status.text)
        corpus = []
        for i in range(len(tweets)): #Data filtering
            review = re.sub(r'\W+',' ',str(tweets[i]))
            review = re.sub(r'https.*(\s)',' ',review)
            review = re.sub(r'https.*',' ',review)
            review = review.lower()
            review = re.sub(r'\s+[a-z]\s+',' ',review)
            review = re.sub(r'^[a-z]\s+',' ',review)
            review = re.sub(r'\s+',' ',review)
            ps = PorterStemmer()
            review_tokenized = nltk.word_tokenize(review)
            review = [ps.stem(word) for word in review_tokenized if word not in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        XX = vectorizer.transform(corpus).toarray()
        with graph.as_default():
            yy_pred = model.predict(XX)
        yy_pred = (yy_pred > 0.5).astype(int)
        y_temp = yy_pred.sum()
        print(y_temp)
        Y_graph.append(y_temp)
        with open('namo_2019.txt','w') as f: #saving the no of +ve comments
            for i,y in enumerate(Y_graph):
                f.write(str(i)+','+str(y)+'\n')
        time.sleep(3600) #Start counting tweets after 1 hour


if __name__ == '__main__':
    print('In Main Function')
    t2 = multiprocessing.Process(target=plo)
    t2.start()
    print('In Main Function')
    t1 = multiprocessing.Process(target=nlp_list)
    t1.start()
    t1.join()
    t2.join()
    time.sleep(2)
    print('In Main Function')
    #t2.join()
    
    




