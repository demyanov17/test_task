import warnings, nltk
import numpy as np
from natasha import NamesExtractor
from tqdm import tqdm
from sklearn.metrics import f1_score
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
nltk.download('punkt')
warnings.filterwarnings("ignore")


def name_recognize_nltk(text):
    prob_thresh = 0.5
    text = str(text)
    result = []
    global morph

    for word in nltk.word_tokenize(text):        
        for p in morph.parse(word):
            if ('Name' in p.tag and p.score >= prob_thresh) or ('Surn' in p.tag and p.score >= prob_thresh):
                result.append([word, p.normal_form, p.score, text])
    
    if len(result) != 0:
        return result
    else:
        return 0

def make_names_list(data):
    
    lst = []
    for l in data[data.role == 'manager'].head(15)[data[
        data.role == 'manager'].new_text.apply(name_recognize_nltk) != 0].NLTK:
        lst += l
    return lst

def get_name_and_replic(data):

    result_names, replics = [], []
    for idx in np.unique(data.dlg_id):
        lst = make_names_list(data[data.dlg_id == idx])
        if len(lst) == 1:
            result_names.append(lst[0][0])
            replics.append(lst[0][3])
        elif len(lst) == 0:
            result_names.append('имя не было указано')
            replics.append('остутсвуют реплики, где менеджер представил себя')
        else:
            add_name = False
            for l in lst:
                if 'зовут' in l[3] or 'компания' in l[3]:
                    result_names.append(l[0])
                    replics.append(l[3])
                    add_name = True
                    break
            if not add_name:
                max_pr, name = 0, '_'
                for l in lst:
                    if max_pr < l[2]:
                        max_pr, name = l[2], l[0]
                result_names.append(name)
                replics.append(l[3])
    return result_names, replics
