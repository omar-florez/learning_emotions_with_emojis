import pandas as pd
from collections import Counter
import ipdb
import numpy as np
import random

np.random.seed(2018)
emoji2classname_all = {
    '😀': '',
    '😁': '',
    '😂': '',
    '😃': '',
    '😄': '',
    '😅': '',
    '😆': '',
    '😇': '',
    '😉': '',
    '😊': 'joy',
    '😋': '',
    '😌': '',
    '😍': '',
    '😎': '',
    '😏': '',
    '😐': '',
    '😑': '',
    '😒': '',
    '😓': '',
    '😔': 'sadness',
    '😕': '',
    '😖': '',
    '😗': '',
    '😘': '',
    '😙': '',
    '😚': '',
    '😛': '',
    '😜': '',
    '😝': '',
    '😞': 'sadness',
    '😟': 'sadness',
    '😠': 'anger',
    '😡': 'anger',
    '😢': '',
    '😣': '',
    '😤': 'anger',
    '😥': '',
    '😦': '',
    '😧': '',
    '😨': '',
    '😩': '',
    '😪': '',
    '😫': '',
    '😬': '',
    '😭': '',
    '😮': '',
    '😯': '',
    '😰': '',
    '😱': '',
    '😲': '',
    '😳': '',
    '😴': '',
    '😵': '',
    '😶': '',
    '😷': '',
    '🙁': 'sadness',
    '🙂': 'joy',
    '🙃': '',
    '🙄': '',
    '🤐': '',
    '🤑': '',
    '🤒': '',
    '🤓': '',
    '🤔': '',
    '🤕': '',
    '🤗': ''
}

#subset of emotions we will work on:
emoji2classname = {k: emoji2classname_all[k] for k in sorted(emoji2classname_all.keys()) if emoji2classname_all[k]!=''}
classname2classid = {k: i for i, k in enumerate(sorted(set(emoji2classname.values())))}
classid2classname = {classname2classid[k]: k for k in classname2classid}
emoji2classid = {k: classname2classid[emoji2classname[k]] for k in emoji2classname}

class EmojiDatasetWords():
    def _init__(self):
        return

    def load_datasets(self):
        saved_objects = np.load('./saved/params/model_params.npy', allow_pickle=True).item()
        self.max_seq_len = saved_objects['max_seq_len']
        self.observations = saved_objects['observations']
        self.emojis = saved_objects['emojis']
        self.labels = saved_objects['labels']
        self.word2idx = saved_objects['word2idx']
        self.idx2word = saved_objects['idx2word']
        self.seq_lengths = saved_objects['seq_lengths']
        self.word_inputs = saved_objects['word_inputs']
        self.idx_train = saved_objects['idx_train']
        self.idx_test = saved_objects['idx_test']
        self.idx_val = saved_objects['idx_val']

        self.x_train = saved_objects['x_train']
        self.y_train = saved_objects['y_train']
        self.seqlen_train = saved_objects['seqlen_train']

        self.x_test = saved_objects['x_test']
        self.y_test = saved_objects['y_test']
        self.seqlen_test = saved_objects['seqlen_test']

        self.x_val = saved_objects['x_val']
        self.y_val = saved_objects['y_val']
        self.seqlen_val = saved_objects['seqlen_val']
        self.trainid_per_labelid = saved_objects['trainid_per_labelid']

        self.vocab_size = saved_objects['vocab_size']
        return

    def write_datasets_tsv(self):
        saved_objects = np.load('./saved/params/model_params.npy', allow_pickle=True).item()
        self.emojis = saved_objects['emojis']
        self.word2idx = saved_objects['word2idx']
        self.idx2word = saved_objects['idx2word']

        train_df = self.decode_dataset(saved_objects['x_train'], saved_objects['y_train'], None, file_path=None)
        test_df = self.decode_dataset(saved_objects['x_test'], saved_objects['y_test'], None, file_path=None)
        val_df = self.decode_dataset(saved_objects['x_val'], saved_objects['y_val'], None, file_path=None)

        train_df.to_csv('./data/train.tsv', sep='\t', index=False)
        test_df.to_csv('./data/test.tsv', sep='\t', index=False)
        val_df.to_csv('./data/val.tsv', sep='\t', index=False)


    def build_dataset(self, file_name, remove_lowfrequent_words=True, max_seq_len=10):
        self.max_seq_len = max_seq_len

        # Read corpus obtained from Twitter
        data_df = pd.read_csv(file_name, sep=',')
        corpus = data_df['observations']
        emojis = data_df['emojis']
        index = np.arange(len(corpus))

        if remove_lowfrequent_words:
            word_counter = Counter(" ".join(corpus).split(" "))
            most_common_words = [word[0] for word in word_counter.most_common(5)]
            temp_corpus = []
            temp_emojis = []
            temp_index = []
            for i, text in zip(index, corpus):
                text_filtered = [word for word in text.split(" ") if word_counter[word] > 20 and word not in most_common_words]
                if len(text_filtered)>0:
                    temp_index.append(i)
                    temp_corpus.append(" ".join(text_filtered))
                    temp_emojis.append(emojis[i])
            corpus = np.array(temp_corpus)
            emojis = np.array(temp_emojis)
            index = np.array(temp_index)

        # Select a smaller number of emojis. Those ones that show more Valence and Arousal
        # Reference: https://www.frontiersin.org/articles/10.3389/fpsyg.2017.00494/full
        self.subset_indices = [i for i in range(len(emojis)) if emojis[i] in emoji2classname.keys()]
        self.observations = corpus[self.subset_indices]
        self.emojis = np.array(emojis[self.subset_indices])
        self.labels = np.array([emoji2classid[emoji] for emoji in self.emojis])

        # Generate a char-based representation of text
        raw_words = {word for text in self.observations for word in text.split(" ")}
        self.word2idx = {' ': 0}
        self.idx2word = {0: ' '}
        for i, w in enumerate(raw_words):
            idx = i+1
            if w == ' ':
                continue
            self.idx2word[idx] = w
            self.word2idx[w] = idx

        self.vocab_size = len(raw_words)

        # Padding short texts with zeros:
        self.word_inputs_ = [[self.word2idx[word] for word in text.split(" ")] for text in self.observations]
        self.seq_lengths = np.array([len(obs_ids) for obs_ids in self.word_inputs_])
        #maximun number of chars per text is equal to the average number of characters in tweets ~ 45
        #self.max_seq_len = int(np.mean(self.seq_lengths))

        #number of chars per observation
        self.word_inputs = np.zeros([len(self.word_inputs_), self.max_seq_len])
        for i in range(len(self.word_inputs)):
            offset = min(self.max_seq_len, len(self.word_inputs_[i]))
            self.word_inputs[i, :offset] = self.word_inputs_[i][:offset]

        # Split dataset:
        n = len(self.labels)
        data_split = (0.8, 0.1, 0.1)
        nTr = int(data_split[0] * n)
        nVal = int(data_split[1] * n)
        nTe = n - nTr - nVal
        idx = list(range(n))
        np.random.shuffle(idx)
        idxTr = idx[:nTr]
        idxVal = idx[nTr:nTr + nVal]
        idxTe = idx[-nTe:]

        self.idx_train, self.idx_test, self.idx_val = idxTr, idxTe, idxVal
        self.x_train = self.word_inputs[idxTr]
        self.y_train = self.labels[idxTr]
        self.seqlen_train = self.seq_lengths[self.idx_train]
        self.x_test = self.word_inputs[idxTe]
        self.y_test = self.labels[idxTe]
        self.seqlen_test = self.seq_lengths[self.idx_test]
        self.x_val = self.word_inputs[idxVal]
        self.y_val = self.labels[idxVal]
        self.seqlen_val = self.seq_lengths[self.idx_val]

        self.trainid_per_labelid = {}
        for train_id in  self.idx_train:
            labelid = self.labels[train_id]
            if labelid not in self.trainid_per_labelid:
                self.trainid_per_labelid[labelid] = []
            self.trainid_per_labelid[labelid].append(train_id)

        print('Number of observations: %d' %n)
        print('Current vocabulary is formed by %d characters' %self.vocab_size)
        print('Maximun number of characters allowed in input: %d' % self.max_seq_len)

        saved_objects = {}
        saved_objects['max_seq_len'] = max_seq_len
        saved_objects['observations'] = self.observations
        saved_objects['emojis'] = self.emojis
        saved_objects['labels'] = self.labels
        saved_objects['vocab_size'] = self.vocab_size

        saved_objects['word2idx'] = self.word2idx
        saved_objects['idx2word'] = self.idx2word

        saved_objects['seq_lengths'] = self.seq_lengths
        saved_objects['word_inputs'] = self.word_inputs

        saved_objects['idx_train'] = self.idx_train
        saved_objects['idx_test'] = self.idx_test
        saved_objects['idx_val'] = self.idx_val

        saved_objects['x_train'] = self.x_train
        saved_objects['y_train'] = self.y_train
        saved_objects['seqlen_train'] = self.seqlen_train
        saved_objects['x_test'] = self.x_test
        saved_objects['y_test'] = self.y_test
        saved_objects['seqlen_test'] = self.seqlen_test
        saved_objects['x_val'] = self.x_val
        saved_objects['y_val'] = self.y_val
        saved_objects['seqlen_val'] = self.seqlen_val
        saved_objects['trainid_per_labelid'] = self.trainid_per_labelid
        np.save('./saved/params/new_model_params.npy', saved_objects)

        #decode training dataset for debudding purposes
        self.decode_dataset(self.x_train, self.y_train, self.emojis[idxTr], file_path='./saved/params/train_data.csv')
        self.decode_dataset(self.x_test, self.y_test, self.emojis[idxTe], file_path='./saved/params/test_data.csv')
        self.decode_dataset(self.x_val, self.y_val, self.emojis[idxVal], file_path='./saved/params/val_data.csv')
        return

    def decode_dataset(self, x, y, emojis, file_path='./saved/results/train_data.csv'):
        corpus = []
        for row in x:
            text = " ".join([self.idx2word[c] for c in row])
            corpus.append(text)

        labels = [classid2classname[yy] for yy in y]

        if emojis is not None:
            data_df = pd.DataFrame({'text': corpus,
                                    'label': labels,
                                    'emojis': emojis})
        else:
            data_df = pd.DataFrame({'text': corpus,
                                    'label': labels})
        if file_path:
            data_df.to_csv(file_path, sep=',')
        return data_df

    def next_train_balanced(self, batch_size=32):
        '''
        Build a mini-batch by randomly picking a number of distinct classes ('width') from the training batch.
        This ensures we ended up with a balanced class distribution in each mini-batch
        :param batch_size: number of observations in the current mini-batch. Yann recommends 32, https://arxiv.org/abs/1804.07612
        :param width: number of distinct classes in the current minibatch
        :return: x, y, and sequence length minibatches
        '''
        assert len(self.idx_train) > batch_size
        #class ids are unique elements
        class_ids = np.unique(self.labels)

        #class_support: number of elements per class
        class_support = batch_size//len(class_ids)
        idx_train_batch = [np.random.choice(self.trainid_per_labelid[labelid], class_support) for labelid in class_ids]

        idx_train_batch = np.array(idx_train_batch).ravel().tolist()
        remaind_len = batch_size - len(idx_train_batch)

        idx_remaind_batch = []
        class_ids_remain = np.random.choice(self.labels, remaind_len)
        for class_id in class_ids_remain:
            idx_train_batch.append(np.random.choice(self.trainid_per_labelid[class_id], 1)[0])

        np.random.shuffle(idx_train_batch)
        x_batch = self.word_inputs[idx_train_batch]
        y_batch = self.labels[idx_train_batch].ravel()
        batch_seqlen = self.seq_lengths[idx_train_batch]

        return x_batch, y_batch, batch_seqlen

    def get_train_test_val_data(self):
        train_data, train_seqlen = self.x_train, self.seqlen_train
        test_data, test_seqlen = self.x_test, self.seqlen_test
        val_data, val_seqlen = self.x_val, self.seqlen_val
        train_label = self.y_train
        test_label = self.y_test
        val_label = self.y_val
        output = [train_data, train_label, train_seqlen,
                  test_data, test_label, test_seqlen,
                  val_data, val_label, val_seqlen,
                  self.vocab_size, self.max_seq_len]
        return output

    def get_idx2char(self):
        return self.idx2word

    def get_label_idx2name(self):
        return self.classid2classname

    def get_vocab_size(self):
        return self.vocab_size

    def get_number_classes(self):
        return len(np.unique(self.labels))

    def get_seq_max_len(self):
        return self.max_seq_len

    def get_class_names(self):
        return [k for k in classname2classid]

