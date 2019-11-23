import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class semEvalDataset(Dataset):
    def __init__(self, folderPath, filePath, tokenizer, seq_len=128, train=True):
        self.train = train
        # load both training and test data
        self.data_train = preprocess_semeval_annotated(folderPath, filePath, train=True, include_tokens=False)
        self.data_test = preprocess_semeval_annotated(folderPath, filePath, train=False, include_tokens=False)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # label encoder for aspect, polarity and joint
        self.aspect_encoder = MultiLabelBinarizer()
        self.polarity_encoder = MultiLabelBinarizer()
        self.joint_encoder = MultiLabelBinarizer()
        self.aspect_encoder.fit(self.data_train.aspect)
        self.polarity_encoder.fit(self.data_train.polarity)
        self.joint_encoder.fit(self.data_train.joint)

        self.aspects_train = self.aspect_encoder.transform(self.data_train.aspect)
        self.aspects_test = self.aspect_encoder.transform(self.data_test.aspect)
        self.polarity_train = self.polarity_encoder.transform(self.data_train.polarity)
        self.polarity_test = self.polarity_encoder.transform(self.data_test.polarity)
        self.joint_train = self.joint_encoder.transform(self.data_train.joint)
        self.joint_test = self.joint_encoder.transform(self.data_test.joint)

    def __len__(self):
        return self.data_train.shape[0] if self.train else self.data_test.shape[0]

    def __getitem__(self, idx):
        if self.train:
            aspect = self.aspects_train[idx]
            polarity = self.polarity_train[idx]
            joint = self.joint_train[idx]
            seq_raw = self.data_train.iloc[idx, 0]
        else:
            aspect = self.aspects_test[idx]
            polarity = self.polarity_test[idx]
            joint = self.joint_train[idx]
            seq_raw = self.data_test.iloc[idx, 0]
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
                  'joint': joint.astype(np.float),'sequence': np.array(seq), 'mask': np.array(mask)}
        return sample


def preprocess_semeval_annotated(folderPath, filePath, train=True, include_tokens=True):
    """
    Builds a dataframe with aspect, polarity, joint and corresponding segment
    for each comment in the annotated SemEval dataset
    :param path: the path to the data
    :param train: return the training set? Otherwise returns test set
    :param include_tokens: include [CLS] and [SEP]?
    :return: pd.Dataframe: a dataframe with columns aspect, polarity, joint and segment
    """

    df = pd.read_csv(os.path.join(folderPath, filePath+('_train.csv' if train else '_test.csv')))
    df['polarity'] = [x.split(',') for x in df['polarity']]
    df['aspect'] = [x.split(',') for x in df['aspect']]
    if include_tokens:
        df['text'] = df['text'].map(lambda x: '[CLS]' + x + '[SEP]')
    df['joint'] = df.apply(lambda x: ['/'.join([a, b]) for a in x['aspect'] for b in x['polarity']], axis=1)
    return df
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        help='Path to the annotated SemEval dataset (should contain semevalLaptops_combi14/16.csv / semevalRestaurants_combi14/16.csv)',
                        required=True)
    args = parser.parse_args()
    data = semEvalDataset('../models/data/semeval/', 'semevalRestaurants_combi16', BertTokenizer.from_pretrained('bert-base-uncased'))

    for i in range(1):
        print(data[i])
