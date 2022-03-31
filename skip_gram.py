import numpy as np
from numpy.linalg import norm
import re, string

class SkipGram():
    def __init__(self, seed, corpus, window=3, N=3, n=0.05, epochs=50):
        np.random.seed(seed)
        self.corpus = corpus
        self.corpus_list = None
        self.window = window
        self.N = N
        self.n = n
        self.epochs = epochs
        self.vocabulary = None
        self.T = 0
        self.V = 0
        self.w_input = None # VxN
        self.w_output = None # NxV
        self.initialize()

    def initialize(self):
        print("Initialize...")
        self.cleanCorpus()
        self.corpus_list = self.corpus.split()
        aux_corpus_list = sorted(self.corpus_list)
        self.vocabulary = []
        for item in aux_corpus_list:
            if item not in self.vocabulary:
                self.vocabulary.append(item)
        self.T = len(self.corpus_list)
        self.V = len(self.vocabulary)
        self.w_input = np.random.uniform(-1, 1, (self.V, self.N))
        self.w_output = np.random.uniform(-1, 1, (self.N, self.V))
        print(self.vocabulary)

    def forwardPropagation(self, word_center):
        # Computing hidden (h) layer
        index = self.vocabulary.index(word_center)
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        x[0][index] = 1
        h = np.dot(x, self.w_input) # 1xN

        # Computing Softmax out layer
        u = np.dot(h, self.w_output) # 1xV
        y_pred = self.softmax(u) # 1xV
        return h, y_pred
    
    def backwardPropagation(self,h, y_pred, word_center, words_contex):
        # Computing sum of prediction errors
        # print(word_center, words_contex)
        index = self.vocabulary.index(word_center)
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        x[0][index] = 1

        sum_error = np.zeros((1, self.V)) # 1xV
        for word_contex in words_contex:
            index = self.vocabulary.index(word_center)
            y_truth = np.zeros((1, self.V)) # one-hot vector, 1xV
            y_truth[0][index] = 1
            e_c = y_pred - y_truth
            sum_error += e_c
        
        # Computing gradient w_input
        multi_prev = np.dot(self.w_output, sum_error.T)
        grad_w_input = np.dot(x.T, multi_prev.T) # VxN
        
        # Computing gradient w_output
        grad_w_output = np.dot(h.T, sum_error) # NxV

        # Updating weight
        self.w_input = self.w_input - self.n * grad_w_input
        self.w_output = self.w_output - self.n * grad_w_output  

    def traing(self):
        for k in range(self.epochs):
            for index, word_center in enumerate(self.corpus_list):
                i = index - self.window
                j = index + self.window

                if i < 0:
                    i = 0
                if j >= self.T:
                    j = self.T
            
                words_contex = self.corpus_list[i:index] + self.corpus_list[index+1:j+1]
                
                h, y_pred = self.forwardPropagation(word_center)
                self.backwardPropagation(h, y_pred, word_center, words_contex)
            
            """ print("Epochs: {}".format(k + 1))
            print(self.w_input)
            print(self.w_output) """
        return self.w_input
    
    def getSimilarity(self, token, head):
        similarities = np.zeros(self.V)
        index = self.vocabulary.index(token)
        u = self.w_input[index]

        for k, v in enumerate(self.w_input):
            similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            similarities[k] = similarity
            # print("Cosine similar = {}, {}".format(similarities[k], self.vocabulary[k]))

        heads = np.argsort(similarities)
        for k in range(head):
            print("Cosine similar = {}, {}".format(similarities[heads[-1*(k+1)]], self.vocabulary[heads[-1*(k+1)]]))
    
    def softmax(self, vector):
        vector = np.exp(vector)
        soft = vector / np.sum(vector)
        return soft

    def cleanCorpus(self):
        self.corpus = self.corpus.strip().lower()
        self.corpus = re.sub("[%s]" % re.escape(string.punctuation), " ", self.corpus)
        # print(corpus)
        # self.corpus = self.corpus.decode("unicode_escape").encode("ascii", "ignore")

if __name__=="__main__":
    corpus = "duct tape work anywher duct tape magic worship"
    corpus = "The man who passes the sentence should swing the sword"
    corpus = "El camino a casa es largo"

    with open("corpus.txt", encoding="utf8") as f:
        corpus = f.read()
        print(type(corpus))

    sg = SkipGram(1234, corpus, window=3, N=10, n=0.05, epochs=500)
    w_input = sg.traing()
    # print((w_input))
    sg.getSimilarity("binarios", 10)

   