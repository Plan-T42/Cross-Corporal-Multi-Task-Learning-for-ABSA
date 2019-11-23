import os
import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class OrganicDataset(Dataset):
    # 0 FOR TRAIN, 1 FOR EVAL, 2 FOR TEST
    def __init__(self, path, tokenizer, seq_len=128, train=0):
        self.train = train
        # load both training and test data
        self.data_train = preprocess_organic_annotated(os.path.join(path, 'organic_train.csv'))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.neg_label_weight = 0.1

        # label encoder for aspect, polarity and joint task
        self.aspect_encoder = MultiLabelBinarizer()
        self.polarity_encoder = MultiLabelBinarizer()
        self.joint_encoder = MultiLabelBinarizer()
        self.aspect_encoder.fit(self.data_train.aspect)
        self.polarity_encoder.fit(self.data_train.polarity)
        self.joint_encoder.fit(self.data_train.joint)

        if self.train == 0:
            self.data = self.data_train
        elif self.train == 1:
            self.data = preprocess_organic_annotated(os.path.join(path, 'organic_eval.csv'))
        else:
            self.data = preprocess_organic_annotated(os.path.join(path, 'organic_test.csv'))

        self.aspects = self.aspect_encoder.transform(self.data.aspect)
        self.polarity = self.polarity_encoder.transform(self.data.polarity)
        self.joint = self.joint_encoder.transform(self.data.joint)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        aspect = self.aspects[idx]
        aspect_weights = np.where(aspect == 1.0, aspect, self.neg_label_weight)
        polarity = self.polarity[idx]
        joint = self.joint[idx]
        joint_weights = np.where(joint == 1.0, joint, self.neg_label_weight)
        seq_raw = self.data.iloc[idx, 0]

        seq = ['[CLS]'] + self.tokenizer.tokenize(seq_raw)[:self.seq_len - 2] + ['[SEP]']
        seq = self.tokenizer.convert_tokens_to_ids(seq)

        # zero-pad sequence and mask
        mask = [1] * len(seq)
        while len(mask) < self.seq_len:
            mask.append(0)
            seq.append(0)

        assert len(mask) == self.seq_len
        assert len(seq) == self.seq_len

        sample = {'aspect': aspect.astype(np.float), 'polarity': polarity.astype(np.float),
                  'joint': joint.astype(np.float), 'sequence': np.array(seq), 'mask': np.array(mask),
                  'aspect_weights': aspect_weights.astype(np.float), 'joint_weights': joint_weights.astype(np.float)}
        return sample


def preprocess_organic_annotated(path):
    data = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    # drop trailing commas in polarity
    data['polarity'] = data.iloc[:, 2].apply(lambda x: x[0])
    data = data.loc[:, ('sequence', 'aspect', 'polarity')]
    # collect all aspects and corresponding polarities for every sentence into lists
    data = data.groupby('sequence', as_index=False).agg({'aspect': list, 'polarity': list})
    data['joint'] = data.apply(lambda x: ['/'.join([a, b]) for a in x['aspect'] for b in x['polarity']], axis=1)
    return data


if __name__ == '__main__':
    data = OrganicDataset('../models/data/organic', BertTokenizer.from_pretrained('bert-base-uncased'))

    for i in range(1):
        print(data[i]['test'])
