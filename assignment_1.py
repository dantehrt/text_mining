import nltk
import numpy as np
import re
import glob
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

from nltk.corpus import wordnet as wn


def get_files(path):
    file_name_list = glob.glob(path)
    print(file_name_list)
    filename_and_document_pairs = []
    for file_name in file_name_list:
        f = open(file_name)
        document = f.read()
        filename_and_document_pairs.append((file_name.split('/')[1], document))
        f.close()

    filename_and_document_pairs.sort()

    return filename_and_document_pairs


def preprocessing_text(text):
    def cleaning_text(text):
        # @の削除
        pattern1 = '@'
        text = re.sub(pattern1, '', text)
        # <b>タグの削除
        pattern2 = '<[^>]*?>'
        text = re.sub(pattern2, '', text)
        # ()内を削除
        pattern3 = '\([^\)]*?\)'
        text = re.sub(pattern3, '', text)
        return text

    def tokenize_text(text):
        text = re.sub('[.,]', '', text)
        return text.split()

    def lemmatize_word(word):
        # make words lower  example: Python =>python
        word = word.lower()

        # lemmatize  example: cooked=>cook
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def remove_stopwords(word, stopwordset):
        if word in stopwordset:
            return None
        else:
            return word

    en_stop = nltk.corpus.stopwords.words('english')

    text = cleaning_text(text)
    tokens = tokenize_text(text)
    tokens = [lemmatize_word(word) for word in tokens]
    tokens = [remove_stopwords(word, en_stop) for word in tokens]
    tokens = [word for word in tokens if word is not None]
    return tokens


def calculate_similarity_based_on_set(filename_and_preprocessed_document_pairs):
    def jaccard_similarity(set_a, set_b):
        # 積集合の要素数を計算
        num_intersection = len(set.intersection(set_a, set_b))
        # 和集合の要素数を計算
        num_union = len(set.union(set_a, set_b))
        # Jaccard係数を算出　空集合の時は1を出力
        try:
            return float(num_intersection) / num_union
        except ZeroDivisionError:
            return 1.0

    def simpson_similarity(set_a, set_b):
        num_intersection = len(set.intersection(set_a, set_b))
        min_num = min(len(set_a), len(set_b))
        try:
            return num_intersection / min_num
        except ZeroDivisionError:
            if num_intersection == 0:
                return 1.0
            else:
                return 0

    def show_heatmap(title, fun):
        print('-----' + title + '係数-----')
        df = pd.DataFrame(columns=[x[0] for x in filename_and_preprocessed_document_pairs])
        length = len(filename_and_preprocessed_document_pairs)
        for i in range(length):
            tmp = []
            for j in range(length):
                tmp.append(fun(set(filename_and_preprocessed_document_pairs[i][1]),
                               set(filename_and_preprocessed_document_pairs[j][1])))
            se = pd.Series(tmp, index=df.columns, name=filename_and_preprocessed_document_pairs[i][0])
            df = df.append(se)
        print(df)

        seaborn.heatmap(df, cmap='viridis')
        plt.title(title)
        plt.ylim(len(filename_and_preprocessed_document_pairs), 0)
        plt.show()

    print('-----集合ベースの類似度計算-----')
    show_heatmap('jaccard', jaccard_similarity)
    show_heatmap('Szymkiewicz-Simpson', simpson_similarity)


def calculate_similarity_based_on_vector(filename_and_preprocessed_document_pairs):
    word2id = {}
    for filename_and_preprocessed_document_pair in filename_and_preprocessed_document_pairs:
        doc = filename_and_preprocessed_document_pair[1]
        for w in doc:
            if w not in word2id:
                word2id[w] = len(word2id)

    def bow_vectorizer():
        filename_and_vector_pairs = []
        for filename_and_preprocessed_document_pair in filename_and_preprocessed_document_pairs:
            doc_vec = [0] * len(word2id)
            filename = filename_and_preprocessed_document_pair[0]
            doc = filename_and_preprocessed_document_pair[1]
            for w in doc:
                doc_vec[word2id[w]] += 1
            filename_and_vector_pairs.append((filename, doc_vec))

        return filename_and_vector_pairs

    def tfidf_vectorizer():
        def tf(word2id, doc):
            term_counts = np.zeros(len(word2id))
            for term in word2id.keys():
                term_counts[word2id[term]] = doc.count(term)
            tf_values = list(map(lambda x: x / sum(term_counts), term_counts))
            return tf_values

        def idf(word2id, docs):
            idf = np.zeros(len(word2id))
            for term in word2id.keys():
                idf[word2id[term]] = np.log(len(docs) / sum([bool(term in doc) for doc in docs]))
            return idf

        filename_and_vector_pairs = []
        docs = [x[1] for x in filename_and_preprocessed_document_pairs]
        for filename_and_preprocessed_document_pair in filename_and_preprocessed_document_pairs:
            filename = filename_and_preprocessed_document_pair[0]
            doc = filename_and_preprocessed_document_pair[1]
            tf_values = tf(word2id, doc)
            idf_values = idf(word2id, docs)
            filename_and_vector_pairs.append((filename, tf_values * idf_values))
        return filename_and_vector_pairs

    def minkowski_distance(list_a, list_b, p=2):
        diff_vec = np.array(list_a) - np.array(list_b)

        return np.linalg.norm(diff_vec, ord=p)

    def minkowski_simirarity(list_a, list_b, p=2):
        return 1 / (1 + minkowski_distance(list_a, list_b, p=2))

    def cosine_similarity(list_a, list_b):
        inner_prod = np.dot(list_a, list_b)
        norm_a = np.linalg.norm(list_a)
        norm_b = np.linalg.norm(list_b)
        return inner_prod / (norm_a * norm_b)

    def show_heatmap(title, fun_vectorization, fun_distance):
        print('-----' + title + '-----')
        filename_and_vector_pairs = fun_vectorization()
        df = pd.DataFrame(columns=[x[0] for x in filename_and_vector_pairs])
        length = len(filename_and_vector_pairs)
        for i in range(length):
            tmp = []
            for j in range(length):
                tmp.append(fun_distance(filename_and_vector_pairs[i][1], filename_and_vector_pairs[j][1]))
            se = pd.Series(tmp, index=df.columns, name=filename_and_vector_pairs[i][0])
            df = df.append(se)

        print(df)
        seaborn.heatmap(df, cmap='viridis')
        plt.title(title)
        plt.ylim(len(filename_and_vector_pairs), 0)
        plt.show()

    print('-----ベクトルベースの類似度計算-----')
    show_heatmap('Bow x euclid', bow_vectorizer, minkowski_simirarity)
    show_heatmap('Bow x cos', bow_vectorizer, cosine_similarity)
    show_heatmap('tf-idf x euclid', tfidf_vectorizer, minkowski_simirarity)
    show_heatmap('tf-idf x cos', tfidf_vectorizer, cosine_similarity)


def main():
    filename_and_document_pairs = get_files('data/*.txt')
    filename_and_preprocessed_document_pairs = [
        (filename_and_document_pair[0], preprocessing_text(filename_and_document_pair[1]))
        for filename_and_document_pair in filename_and_document_pairs
    ]
    calculate_similarity_based_on_set(filename_and_preprocessed_document_pairs)
    calculate_similarity_based_on_vector(filename_and_preprocessed_document_pairs)


if __name__ == '__main__':
    main()
