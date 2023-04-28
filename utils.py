from collections import defaultdict
import sys, string, json, os
from collections import Counter

class naiveBayes:
    def __init__(self):
        # self.STOPWORDS = Counter()
        self.prior = defaultdict(int)
        self.countwc = defaultdict(lambda : defaultdict(int))
        self.countc = defaultdict(int)
        self.vocab = Counter()
        self.total_sentences = 0
        self.wordsexp = Counter()

    def clean(self, s):
        s = s.translate(str.maketrans('','',string.punctuation))
        s = s.translate(str.maketrans('','','1234567890'))
        s = s.lower()
        return s

    def train(self, path):
        for folderpn in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
            path_pn = os.path.join(path, folderpn)
            class1 = folderpn.split('_')[0]

            for foldertd in [x for x in os.listdir(path_pn) if os.path.isdir(os.path.join(path_pn, x))]:
                path_td = os.path.join(path_pn, foldertd)
                class2 = foldertd.split('_')[0]

                for fold in [x for x in os.listdir(path_td) if os.path.isdir(os.path.join(path_td, x))]:
                    path_fold = os.path.join(path_td, fold)

                    for file in os.listdir(path_fold):
                        file_path = os.path.join(path_fold, file)

                        with open(file_path) as f:
                            for sent in f.readlines():
                                self.total_sentences+=1
                                self.prior[class1]+=1
                                self.prior[class2]+=1

                                sent = self.clean(sent)

                                for word in sent.split(' '):
                                    # if word not in self.STOPWORDS:
                                    self.vocab[word]+=1
                                    self.countwc[class1][word]+=1
                                    self.countwc[class2][word]+=1
                                    self.countc[class1]+=1
                                    self.countc[class2]+=1

        for classname in self.prior:
            self.prior[classname]/=self.total_sentences

        print(len(self.vocab))
        print(set([x[0] for x in self.vocab.most_common(int(len(self.vocab) * 0.02))]))
        # with open('nbmodel.txt', 'w') as f:
        #     json.dump({'Prior':self.prior, 'Countwc':self.countwc, 'Countc':self.countc, 'V':len(self.vocab), 'STOPWORDS':list(self.STOPWORDS)}, f, indent=2)


if __name__=="__main__":
    naiveBayes().train(r'D:\USC\Study_Material\Spring_22\NLP\prog\p1\op_spam_training_data')