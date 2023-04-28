from collections import defaultdict
import sys, string, json, os

def naiveBayesTrain(path):
    
    #taken from nltk corpus
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 
    
    #P(c|d) prop to P(c) * P(w1|c) * P(w2|c) * .....
    prior = defaultdict(int)
    likelihood = defaultdict(lambda : defaultdict(int))
    class_counts = defaultdict(int)
    vocab = set()
    total_sent = 0

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
                    prior[class1]+=1
                    prior[class2]+=1
                    total_sent+=1
                    sent = open(file_path).readlines()[0].translate(str.maketrans('','',string.punctuation)).translate(str.maketrans('','','1234567890'))
                    for word in sent.split(' '):
                        word = word.strip().lower()
                        if word in stopwords:
                            continue
                        vocab.add(word)
                        likelihood[class1][word]+=1
                        likelihood[class2][word]+=1
                        class_counts[class1]+=1
                        class_counts[class2]+=1

    for classname in prior:
        prior[classname]/=total_sent

    with open('nbmodel.txt', 'w') as f:
        json.dump({'Prior':prior, 'Likelihood':likelihood, 'ClassCounts':class_counts, 'Vocabulary':list(vocab)}, f, indent=2)


naiveBayesTrain(sys.argv[1])