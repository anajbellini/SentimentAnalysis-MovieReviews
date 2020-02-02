"""
ALGORITMO: Support Vector Machine
REPRESENTACAO DE TEXTO: Bag-of-Words
"""

import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from models.text_processing import TextProcessing

# inicio do algoritmo
print(str(datetime.datetime.now()))

# importacao do dataset
file_train = pd.read_csv('datasets/imdb-train.csv').drop("PhraseId", axis=1)
file_test = pd.read_csv('datasets/imdb-test-labelled.csv').drop("PhraseId", axis=1)
dataset = file_train.append(file_test, ignore_index=True)

# esse codigo sera executado 10x, entao usarei essa variavel como seed
iteracao = 0

# embaralhar dataset (pegar 10k exemplos apos o shuffle)
dataset = dataset.sample(n=10000, random_state=iteracao).reset_index(drop=True)

# processamento do texto
corpus = []
processor = TextProcessing(reduction='S')
# processor = TextProcessing(reduction='L')

print('Processando os textos...')
for sentence in range(len(dataset)):
    processed_sentence = processor.process_text(dataset['Phrase'][sentence])
    corpus.append(processed_sentence)

# representacao em vetores e obtencao das classes
print('Gerando o Bag-of-Words...')
bag_of_words = processor.generate_bagofwords(corpus)
classes = dataset.iloc[:, -1].values

# desconsiderando palavras no BoW que aparecem 1x
bag_of_words = bag_of_words[:, bag_of_words.sum(axis=0) > 1]

# k-Fold cross validation
k_fold = KFold(n_splits=10, random_state=iteracao, shuffle=True)
aux_accuracy = 0
aux_f1_score = 0
aux_precision = 0
aux_recall = 0
conf_matrices = np.zeros((2, 2))
i = 1

# treino, teste e avaliacao
print('Iniciando o k-Fold...')
for train_index, test_index in k_fold.split(bag_of_words):
    x_train, x_test = bag_of_words[train_index], bag_of_words[test_index]
    y_train, y_test = classes[train_index], classes[test_index]

    # treino do modelo
    print(str(datetime.datetime.now()))
    print(f'Treinando o Modelo {i}...')
    classifier = SVC(kernel='linear', gamma=0.1, random_state=iteracao).fit(x_train, y_train)
    print(f'Treino do Modelo {i} finalizado.')

    # classificando o conjunto de teste
    y_pred = classifier.predict(x_test)

    # metricas de desempenho
    aux_accuracy += accuracy_score(y_test, y_pred)
    aux_f1_score += f1_score(y_test, y_pred)
    aux_precision += precision_score(y_test, y_pred)
    aux_recall += recall_score(y_test, y_pred)
    conf_matrices += np.asarray(confusion_matrix(y_test, y_pred))

    i += 1

# resultados
print(f'ITERATION #{iteracao} -----------------------')
print(f'Accuracy = {aux_accuracy / k_fold.n_splits}')
print(f'F1 Score = {aux_f1_score / k_fold.n_splits}')
print(f'Precision = {aux_precision / k_fold.n_splits}')
print(f'Recall = {aux_recall / k_fold.n_splits}')
print(f'Examples x Attributes = {bag_of_words.shape}')

# soma das matrizes de confusao (sao 10)
print(f'Confusion Matrix = \n{np.array(list(conf_matrices))}')

print(str(datetime.datetime.now()))
