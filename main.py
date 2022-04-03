from numpy.linalg import norm

import numpy as np
import matplotlib.pyplot as plt
import re, string

from plot import plot

class Word2Vec():
    def __init__(self, seed, corpus, kind="sg", window=3, N=3, n=0.05, epochs=50):
        """
        seed:
        corpus:
        king: sg or cbow
        """
        np.random.seed(seed)
        self.corpus = corpus
        self.kind = kind
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
        self.clean_corpus()
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

    def forward_propagation_sg(self, target_word_index):
        # Computing hidden (h) layer
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        x[0][target_word_index] = 1
        h = np.dot(x, self.w_input) # 1xN

        # Computing Softmax out layer
        u = np.dot(h, self.w_output) # 1xV
        y_pred = self.softmax(u) # 1xV
        return h, y_pred
    
    def backward_propagation_sg(self,h, y_pred, target_word_index, context_words_index):
        # Computing sum of prediction errors
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        x[0][target_word_index] = 1

        sum_ec = np.zeros((1, self.V)) # 1xV
        for context_word_index in context_words_index:
            y_truth = np.zeros((1, self.V)) # one-hot vector, 1xV
            y_truth[0][context_word_index] = 1
            ec = y_pred - y_truth
            sum_ec += ec
        # print(sum_ec)
        
        # Computing gradient w_input
        multi_prev = np.dot(self.w_output, sum_ec.T)
        grad_w_input = np.dot(x.T, multi_prev.T) # VxN
        
        # Computing gradient w_output
        grad_w_output = np.dot(h.T, sum_ec) # NxV

        # Updating weight
        self.w_input = self.w_input - self.n * grad_w_input
        self.w_output = self.w_output - self.n * grad_w_output

    def forward_propagation_cbow(self, context_words_index):
        # Computing hidden (h) layer
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        for context_word_index in context_words_index:
            x_aux = np.zeros((1, self.V)) # one-hot vector, 1xV
            x_aux[0][context_word_index] = 1
            x += x_aux
        x = x / len(context_words_index)
        h = np.dot(x, self.w_input) # 1xN

        # Computing Softmax out layer
        u = np.dot(h, self.w_output) # 1xV
        y_pred = self.softmax(u) # 1xV
        return h, y_pred

    def forward_propagation_cbow(self,h, y_pred, target_word_index, context_words_index):
        # Computing sum of prediction errors
        x = np.zeros((1, self.V)) # one-hot vector, 1xV
        for context_word_index in context_words_index:
            x_aux = np.zeros((1, self.V)) # one-hot vector, 1xV
            x_aux[0][context_word_index] = 1
            x += x_aux
        x = x / len(context_words_index)

        y_truth = np.zeros((1, self.V)) # one-hot vector, 1xV
        y_truth[0][target_word_index] = 1
        ec = y_pred - y_truth
        
        # Computing gradient w_input
        multi_prev = np.dot(self.w_output, ec.T)
        grad_w_input = np.dot(x.T, multi_prev.T) # VxN
        
        # Computing gradient w_output
        grad_w_output = np.dot(h.T, ec) # NxV

        # Updating weight
        self.w_input = self.w_input - self.n * grad_w_input
        self.w_output = self.w_output - self.n * grad_w_output  

    def traing(self):
        for _ in range(self.epochs):
            for k, target_word in enumerate(self.corpus_list):
                target_word_index = self.vocabulary.index(target_word)
                i = k - self.window
                j = k + self.window

                if i < 0:
                    i = 0
                if j >= self.T:
                    j = self.T
            
                context_words = self.corpus_list[i:k] + self.corpus_list[k+1:j+1]
                context_words_index = []
                for context_word in context_words:
                    index = self.vocabulary.index(context_word)
                    context_words_index.append(index)
                
                h, y_pred = self.forward_propagation_sg(target_word_index)
                self.backward_propagation_sg(h, y_pred, target_word_index, context_words_index)

                """ h, y_pred = self.forward_propagation_cbow(context_words_index)
                self.forward_propagation_cbow(h, y_pred, target_word_index, context_words_index) """
    
    def get_similarity(self, token, head):
        data = np.zeros((head, self.N))
        rows_name = []

        similarities = np.zeros(self.V)
        if token in self.vocabulary:
            index = self.vocabulary.index(token)
            u = self.w_input[index]

            for k, v in enumerate(self.w_input):
                similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                similarities[k] = similarity
                # print("Cosine similar = {}, {}".format(similarities[k], self.vocabulary[k]))

            heads = np.argsort(similarities)
            for k in range(head):
                print("Cosine similar = {}, {}".format(similarities[heads[-1*(k+1)]], self.vocabulary[heads[-1*(k+1)]]))
                data[k] = self.w_input[heads[-1*(k+1)]]
                rows_name.append(self.vocabulary[heads[-1*(k+1)]])
        else:
            print("Not found..")
        return data, rows_name

    def softmax(self, vector):
        vector = np.exp(vector)
        soft = vector / np.sum(vector)
        return soft

    def clean_corpus(self):
        self.corpus = self.corpus.strip().lower()
        self.corpus = self.corpus.replace("«", "")
        self.corpus = self.corpus.replace("»", "")
        self.corpus = self.corpus.replace("\u200b", "")
        self.corpus = self.corpus.replace("—", "")

        self.corpus = re.sub("[%s]" % re.escape(string.punctuation), " ", self.corpus)
        # print(corpus)
        # self.corpus = self.corpus.decode("unicode_escape").encode("ascii", "ignore")
    
    def save_model(self):
        np.savez("go_sg.npz", self.w_input, self.w_output, self.vocabulary)

    def load_model(self):
        weight = np.loadz("go_sg.npz")

if __name__=="__main__":
    corpus = "duct tape work anywher duct tape magic worship"
    corpus = "The man who passes the sentence should swing the sword"
    corpus = "El camino a casa es largo"

    with open("corpus.txt", encoding="utf8") as f:
        corpus = f.read()

    w2v = Word2Vec(1234, corpus, window=3, N=15, n=0.05, epochs=10)
    w2v.traing()

    while True:
        token = input("Ingresa token: ")
        if token == "exit":
            break
        
        data, rows_name = w2v.get_similarity(token, 10)
        plot(data, rows_name)   