"""
ALGORITMO: Naive Bayes
REPRESENTACAO DE TEXTO: TF-IDF
"""

import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from models.text_processing import TextProcessing

# importacao do dataset do IMDb
# file_train = pd.read_csv(
#     'C:\\Users\\anaju\\PycharmProjects\\SentimentAnalysis-MovieReviews\\datasets\\imdb-train.csv'
# ).drop("PhraseId", axis=1)
# file_test = pd.read_csv(
#     'C:\\Users\\anaju\\PycharmProjects\\SentimentAnalysis-MovieReviews\\datasets\\imdb-test-labelled.csv'
# ).drop("PhraseId", axis=1)
# dataset = file_train.append(file_test, ignore_index=True)

# importacao do dataset do Rotten Tomatoes
dataset = pd.read_csv(
    'C:\\Users\\anaju\\PycharmProjects\\SentimentAnalysis-MovieReviews\\datasets\\rotten-tomatoes.tsv',
    sep='\t', encoding='ISO-8859–1'
)
dataset = dataset.drop(['id', 'rating', 'critic', 'top_critic', 'publisher', 'date'], axis=1)
dataset.dropna(inplace=True)
dataset = dataset.reset_index(drop=True)
dataset['fresh'] = dataset['fresh'].replace({'rotten': 0, 'fresh': 1})

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
    # processed_sentence = processor.process_text(dataset['Phrase'][sentence])   # IMDb
    processed_sentence = processor.process_text(dataset['review'][sentence])  # Rotten Tomatoes
    corpus.append(processed_sentence)

# representacao em vetores e obtencao das classes
print('Gerando o TF-IDF...')
tf_idf = processor.generate_tfidf(corpus)
classes = dataset.iloc[:, -1].values

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
for train_index, test_index in k_fold.split(tf_idf):
    x_train, x_test = tf_idf[train_index], tf_idf[test_index]
    y_train, y_test = classes[train_index], classes[test_index]

    # treino do modelo
    print(f'Gerando o Modelo {i}...')
    classifier = MultinomialNB().fit(x_train, y_train)

    # classificando o conjunto de teste
    y_pred = classifier.predict(x_test)

    # metricas de desempenho
    aux_accuracy += accuracy_score(y_test, y_pred)
    aux_f1_score += f1_score(y_test, y_pred)
    aux_precision += precision_score(y_test, y_pred)
    aux_recall += recall_score(y_test, y_pred)
    conf_matrices += np.asarray(confusion_matrix(y_test, y_pred))

    print(f'Modelo {i} finalizado e avaliado.')
    i += 1

# resultados
print(f'\nITERATION #{iteracao} -----------------------')
print(f'Accuracy = {aux_accuracy / k_fold.n_splits}')
print(f'F1 Score = {aux_f1_score / k_fold.n_splits}')
print(f'Precision = {aux_precision / k_fold.n_splits}')
print(f'Recall = {aux_recall / k_fold.n_splits}')
print(f'Examples x Attributes = {tf_idf.shape}')
print(f'Confusion Matrix = \n{np.array(list(conf_matrices))}')
