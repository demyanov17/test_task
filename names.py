def name_recognize_nltk(text):
    prob_thresh = 0.5
    text = str(text)
    result = ''
    global morph

    for word in nltk.word_tokenize(text):        
        for p in morph.parse(word):
            if ('Name' in p.tag and p.score >= prob_thresh) or ('Surn' in p.tag and p.score >= prob_thresh):
                result += '{:<12}({:>12})score:{:0.3}'.format(word, p.normal_form, p.score)
                result += ' \ '
    
    if result != '':
        return result
    else:
        return 0