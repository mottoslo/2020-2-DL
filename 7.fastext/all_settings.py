from gensim.models import FastText
from gensim.models.word2vec import PathLineSentences
import time

#model 4
sentences = PathLineSentences("C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\train_data\\")
model = FastText(sentences=sentences, size=300, window=5, min_count=10, workers=8, sg=1, hs=0,
                  negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
                 word_ngrams=1, min_n=3, max_n=6)
start = time.time()
model.save("fastText4.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(f'모델4 학습에 걸린 시간 : {time.time() - start}')

ev = time.time()
model = FastText.load("fastText4.model")
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(model.wv.most_similar("thank____you", topn=20))
print(len(model.wv.vocab))
print(f'모델4 평가에 걸린 시간 : {time.time() - ev}')

#model 2
sentences = PathLineSentences("C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\train_data\\")
model = FastText(sentences=sentences, size=100, window=5, min_count=10, workers=8, sg=1, hs=0,
                  negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
                 word_ngrams=1, min_n=3, max_n=6)
start = time.time()
model.save("fastText2.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(f'모델2 학습에 걸린 시간 : {time.time() - start}')

ev = time.time()
model = FastText.load("fastText2.model")
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(model.wv.most_similar("thank____you", topn=20))
print(len(model.wv.vocab))
print(f'모델2 평가에 걸린 시간 : {time.time() - ev}')

#model 3
sentences = PathLineSentences("C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\train_data\\")
model = FastText(sentences=sentences, size=300, window=5, min_count=10, workers=8, sg=1, hs=0,
                  negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
                 word_ngrams=1, min_n=2, max_n=3)
start = time.time()
model.save("fastText3.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(f'모델3 학습에 걸린 시간 : {time.time() - start}')

ev = time.time()
model = FastText.load("fastText3.model")
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(model.wv.most_similar("thank____you", topn=20))
print(len(model.wv.vocab))
print(f'모델3 평가에 걸린 시간 : {time.time() - ev}')

#model 1
sentences = PathLineSentences("C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\train_data\\")
model = FastText(sentences=sentences, size=100, window=5, min_count=10, workers=8, sg=1, hs=0,
                  negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
                 word_ngrams=1, min_n=2, max_n=3)
start = time.time()
model.save("fastText1.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(f'모델1 학습에 걸린 시간 : {time.time() - start}')

ev = time.time()
model = FastText.load("fastText1.model")
score, predictions = model.wv.evaluate_word_analogies('C:\\Users\\boss\\Desktop\\20-2\\딥러닝\\assignments\\과제7\\data\\questions-words.txt')
print(score)
print(model.wv.most_similar("thank____you", topn=20))
print(len(model.wv.vocab))
print(f'모델1 평가에 걸린 시간 : {time.time() - ev}')
