import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class TextProcessing:
    """
    Métodos para realizar o processamento de um texto.

    Existem 4 pipelines diferentes:
    1. caixa baixa -> pontuações -> stop words -> stemming (Porter) -> Bag of Words (TF)
    2. caixa baixa -> pontuações -> stop words -> stemming (Porter) -> TF-IDF
    3. caixa baixa -> pontuações -> stop words -> POS Tagging -> lematizacao -> Bag of Words (TF)
    4. caixa baixa -> pontuações -> stop words -> POS Tagging -> lematizacao -> TF-IDF

    NOTA: pretendo reorganizar isso tudo com a classe sklearn.pipeline.Pipeline algum dia. Descobri/estudei sobre ela
    somente depois que meu trabalho já estava finalizado.
    """

    def __init__(self, reduction):
        self.reduction = reduction

    def process_text(self, sentence):
        processed_sentence = []

        def lowercase():
            """
            Método para conversão do texto em caixa baixa.
            """
            return sentence.lower()

        def remove_punctuation(lower_sentence):
            """
            Método para remover pontuações do texto, incluindo tags HTML.
            """
            return lower_sentence.replace('<br />', ' ').translate(
                lower_sentence.maketrans(string.punctuation, ' ' * len(string.punctuation))
            )

        def remove_stopwords(punctuation_sentence):
            """
            Método para remover stop words do texto.
            """
            tokens = word_tokenize(punctuation_sentence)
            stopwords_list = set(stopwords.words('english'))
            return [word for word in tokens if word not in stopwords_list]

        def lemmatization_pos(stopwords_texts):
            """
            Método que realiza a lematizacao de cada token do texto.
            """

            def pos_wordnet(pos_tags):
                """
                Método que converte tags do POS para tags do Wordnet.
                """
                if pos_tags.startswith('J'):
                    return wordnet.ADJ
                elif pos_tags.startswith('V'):
                    return wordnet.VERB
                elif pos_tags.startswith('N'):
                    return wordnet.NOUN
                elif pos_tags.startswith('R'):
                    return wordnet.ADV
                else:
                    return None

            lemmatizer = WordNetLemmatizer()
            pos_tagging = pos_tag(stopwords_texts)
            lemmatizer_tags = map(lambda x: (x[0], pos_wordnet(x[1])), pos_tagging)

            lemmatized_sentence = []
            for word, tag in lemmatizer_tags:
                if tag is None:
                    lemmatized_sentence.append(word)
                else:
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

            return " ".join(lemmatized_sentence)

        def stemming(stopwords_texts) -> object:
            """
            Método que realiza o stemming de cada token do texto.
            """
            stemmer = PorterStemmer()
            stemmed_sentence = [stemmer.stem(word) for word in stopwords_texts]
            return " ".join(stemmed_sentence)

        if self.reduction in ('S', 'L'):
            texts = lowercase()
            texts = remove_punctuation(texts)
            texts = remove_stopwords(texts)
            processed_sentence = stemming(texts) if self.reduction == 'S' else lemmatization_pos(texts)
        else:
            print('ERRO: tipo de processamento escolhido é inválido. Opções disponíveis são "S" ou "L".')
        return processed_sentence

    @staticmethod
    def generate_bagofwords(processed_dataset):
        """
        Método para transformar o texto em Bag-of-Words, ou TF.
        """
        vectorizer = CountVectorizer()
        bag_of_words = vectorizer.fit_transform(processed_dataset).toarray()
        return bag_of_words

    @staticmethod
    def generate_tfidf(processed_dataset):
        """
        Método para transformar o texto em vetores com TF-IDF.
        """
        vectorizer_tf_idf = TfidfVectorizer()
        tf_idf = vectorizer_tf_idf.fit_transform(processed_dataset).toarray()
        return tf_idf
