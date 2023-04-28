from collections import defaultdict
from math import log
import sys, string, json, os


class naiveBayes:
    def __init__(self):     
        with open('nbmodel.txt') as mod:
            model = json.load(mod)
            self.prior = model['Prior']
            self.countwc = model['Countwc']
            self.countc = model['Countc']
            self.STOPWORDS = set(model['STOPWORDS'])
            self.V = model['V']

    def clean(self, s):
        s = s.lower().replace('.',' ').replace(',',' ').replace('&',' ').replace('/',' ').replace('-','')
        return s

    def classifier(self, path):

        with open('nboutput.txt', 'w') as out:

            for folderpn in [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]:
                path_pn = os.path.join(path, folderpn)

                for foldertd in [x for x in os.listdir(path_pn) if os.path.isdir(os.path.join(path_pn, x))]:
                    path_td = os.path.join(path_pn, foldertd)

                    for fold in [x for x in os.listdir(path_td) if os.path.isdir(os.path.join(path_td, x))]:
                        path_fold = os.path.join(path_td, fold)

                        for file in os.listdir(path_fold):
                            file_path = os.path.join(path_fold, file)

                            with open(file_path) as f:

                                prob_positive, prob_negative, prob_truthful, prob_deceptive = log(self.prior['positive']), log(self.prior['negative']), log(self.prior['truthful']), log(self.prior['deceptive'])
                                
                                for sent in f.readlines():
                                    sent = self.clean(sent)

                                    for word in sent.split(' '):
                                        if word not in self.STOPWORDS:
                                            word = word.rstrip('\'\"-,.:;!?() ').lstrip('\'\"-,.:;!?()').strip('\n').strip('!')
                                            word = re.sub(r'\d', '', word)
                                            if word in self.countwc['positive']:
                                                prob_positive+=log((self.countwc['positive'][word] + 1) / (self.countc['positive'] + self.V))
                                            else:
                                                prob_positive+=log(1 / (self.countc['positive'] + self.V))

                                            if word in self.countwc['negative']:
                                                prob_negative+=log((self.countwc['negative'][word] + 1) / (self.countc['negative'] + self.V))
                                            else:
                                                prob_negative+=log(1 / (self.countc['negative'] + self.V))

                                            if word in self.countwc['truthful']:
                                                prob_truthful+=log((self.countwc['truthful'][word] + 1) / (self.countc['truthful'] + self.V))
                                            else:
                                                prob_truthful+=log(1 / (self.countc['truthful'] + self.V))

                                            if word in self.countwc['deceptive']:
                                                prob_deceptive+=log((self.countwc['deceptive'][word] + 1) / (self.countc['deceptive'] + self.V))
                                            else:
                                                prob_deceptive+=log(1 / (self.countc['deceptive'] + self.V))

                                out.write(f'{"dtercuetphtfiuvle"[int(prob_truthful>prob_deceptive):17:2] } {"npeogsaittiivvee"[int(prob_positive>prob_negative ):16:2]} {file_path}\n')



if __name__=="__main__":
    naiveBayes().classifier(sys.argv[1])