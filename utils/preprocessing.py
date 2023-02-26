from collections import Counter
from itertools import chain
import numpy as np
import tarfile
import zipfile
import os
import urllib.request as urldownload
import re
import spacy
import string
import torch


def get_data():
    root = './data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz'
    filename = '20_newsgroups.tar.gz'
    dpath = os.path.join(root, '20_newsgroups')
    x, y = [], []
    tpath = os.path.join(root, filename)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.isfile(tpath):
        print('Downloading %s' % (filename))
        urldownload.urlretrieve(url, tpath)
    else:
        print('%s already exists' % (filename))
    with tarfile.open(tpath, 'r') as tfile:
        print('Extracting %s' % (filename))
        def is_within_directory(directory, target):
        	
        	abs_directory = os.path.abspath(directory)
        	abs_target = os.path.abspath(target)
        
        	prefix = os.path.commonprefix([abs_directory, abs_target])
        	
        	return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
        	for member in tar.getmembers():
        		member_path = os.path.join(path, member.name)
        		if not is_within_directory(path, member_path):
        			raise Exception("Attempted Path Traversal in Tar File")
        
        	tar.extractall(path, members, numeric_owner=numeric_owner) 
        	
        
        safe_extract(tfile, root)
    print('Reading files')
    for genre in os.listdir(dpath):
        if genre[0] != '.':
            gpath = os.path.join(dpath, genre)
            for file in os.listdir(gpath):
                if file[0] != '.':
                    with open(os.path.join(gpath, file), encoding='latin-1') as f:
                        t = f.read()
                        f.close()
                        lines = t.split("\n")
                        nlines = len(lines)
                        start = 25 if nlines > 25 else nlines-1
                        for i in range(start, -1, -1):
                            matches = 0
                            line = lines[i].lower()
                            for g in genre.split('.'):
                                if ((g in line) & (':' in line)) | ('@' in line) | line.startswith('path') | line.startswith('xref'):
                                    matches = + 1
                            if matches > 0:
                                lines.pop(i)
                        t = "\n".join(lines)
                        x.append(t)
                        y.append(genre)
    return(x, y)


def get_embedding_index(embedding_dim):
    root = './data'
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    filename = 'glove.6B.zip'
    dpath = os.path.join(root, 'glove.6B')
    embeddings_index = {}
    tpath = os.path.join(root, filename)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.isfile(tpath):
        print('Downloading %s' % (filename))
        urldownload.urlretrieve(url, tpath)
    else:
        print('%s already exists' % (filename))
    with zipfile.ZipFile(tpath, 'r') as zfile:
        print('Extracting %s' % (filename))
        zfile.extractall(dpath)
    with open(os.path.join(dpath, 'glove.6B.' + str(embedding_dim) + 'd.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


class TwentyNewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Vectorizer():
    def __init__(self, max_len=None, n_words=None, normalize=False, pad_token='<pad>', oov_token='<unk>'):
        self.n_words = n_words
        self.max_len = max_len
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.normalize = normalize
        self.fitted_max_length = None
        if normalize:
            self._nlp = spacy.load('en_core_web_sm')
        else:
            self._stopwords = spacy.load('en_core_web_sm').Defaults.stop_words

    def fit(self, X):
        X = self._tokenize(X)
        self._build_vocab(X)
        if self.max_len is None:
            lengths = []
            for text in self._vectorize(X):
                lengths.append(len(text))
            self.fitted_max_length = np.max(lengths)
        return self

    def transform(self, X):
        X = self._vectorize(self._tokenize(X))
        return self._pad(X, max_len=self.fitted_max_length if self.max_len is None else self.max_len)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _tokenize(self, X):
        sequences = []
        for text in X:
            sequence = []
            if self.normalize:
                for token in self._nlp(text):
                    if (token.is_alpha) & (not token.is_stop) & (token.pos_ != 'PRON'):
                        sequence.append(token.lemma_)
                    else:  # Prevent empty tensors
                        sequence.append(self.oov_token)
            else:
                for token in self._clean_string(text).split(" "):
                    # Getting rid of the english stop words and the tokens of two or one letter length
                    if (token not in self._stopwords) & (token not in string.punctuation) & (len(token) > 1):
                        sequence.append(token)
                    else:  # Prevent empty tensors
                        sequence.append(self.oov_token)
            sequences.append(sequence)
        return sequences

    def _clean_string(self, X):
        X = re.sub(r"[\W_0-9]+", " ", X)
        X = re.sub(r"\'s", " \'s", X)
        X = re.sub(r"\'d", " \'d", X)
        X = re.sub(r"\'re", " \'re", X)
        X = re.sub(r"n\'t", " n\'t", X)
        X = re.sub(r"\'ve", " have", X)
        X = re.sub(r",", " , ", X)
        X = re.sub(r"!", " ! ", X)
        X = re.sub(r":", " : ", X)
        X = re.sub(r"\(", " \( ", X)
        X = re.sub(r"\)", " \) ", X)
        X = re.sub(r"{", " } ", X)
        X = re.sub(r"{", " } ", X)
        X = re.sub(r"\?", " \? ", X)
        X = re.sub(r"<br />", "", X)
        X = re.sub(r"\s\w\s", " ", X)
        X = re.sub(r"^\w\s", " ", X)
        X = re.sub(r"\s\w$", " ", X)
        X = re.sub(r"\s{2,}", " ", X)
        return X.lower().strip()

    def _build_vocab(self, X):
        self.word_counts = Counter(chain(*X))
        word_list = self.word_counts.most_common()
        if self.oov_token in self.word_counts.keys():
            del self.word_counts[self.oov_token]
        word_list = self.word_counts.most_common()
        word_list.insert(0, (self.pad_token, 0))
        word_list.insert(1, (self.oov_token, 0))
        if self.n_words is None:
            self.vocab_size = len(word_list)
        else:
            word_list = word_list[:self.n_words + 2]
            self.vocab_size = len(word_list)
        self.word_index = {w: i for i, (w, _) in enumerate(word_list, start=0)}
        self.index_word = {i: w for w, i in self.word_index.items()}

    def _vectorize(self, X):  # pad_sequence requires a list of Tensors
        if self.max_len:
            return [torch.LongTensor([self.word_index[word] if word in self.word_index.keys() else self.word_index[self.oov_token]
                                     for word in text][:self.max_len]) for text in X]
        else:
            return [torch.LongTensor([self.word_index[word] if word in self.word_index.keys() else self.word_index[self.oov_token]
                                     for word in text]) for text in X]

    def _pad(self, X, max_len):
        dim0 = len(X)
        tensor = torch.zeros([dim0, max_len], dtype=torch.int64)
        for i in range(dim0):
            seq_len = X[i].size(0)
            length = seq_len if (seq_len < max_len) else max_len
            tensor[i][:length] = X[i][:length]
        return tensor
