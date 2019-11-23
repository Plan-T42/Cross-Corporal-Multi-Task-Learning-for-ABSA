import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class TripadvisorDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=128, train=True):
        self.train = train
        # load both training and test data
        self.data_train = preprocess_tripad_annotated(path, train=True, include_tokens=False)
        self.data_test = preprocess_tripad_annotated(path, train=False, include_tokens=False)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # label encoder for aspect, polarity and joint task
        self.aspect_encoder = MultiLabelBinarizer()
        self.polarity_encoder = MultiLabelBinarizer()
        self.joint_encoder = MultiLabelBinarizer()
        self.aspect_encoder.fit(self.data_train.aspects)
        self.polarity_encoder.fit(self.data_train.polarities)
        self.joint_encoder.fit(self.data_train.joint)

        self.aspects_train = self.aspect_encoder.transform(self.data_train.aspects)
        self.aspects_test = self.aspect_encoder.transform(self.data_test.aspects)
        self.polarity_train = self.polarity_encoder.transform(self.data_train.polarities)
        self.polarity_test = self.polarity_encoder.transform(self.data_test.polarities)
        self.joint_train = self.joint_encoder.transform(self.data_train.joint)
        self.joint_test = self.joint_encoder.transform(self.data_test.joint)

    def __len__(self):
        return self.data_train.shape[0] if self.train else self.data_test.shape[0]

    def __getitem__(self, idx):
        if self.train:
            aspect = self.aspects_train[idx]
            polarity = self.polarity_train[idx]
            joint = self.joint_train[idx]
            seq_raw = self.data_train.iloc[idx, 3]
        else:
            aspect = self.aspects_test[idx]
            polarity = self.polarity_test[idx]
            joint = self.joint_test[idx]
            seq_raw = self.data_test.iloc[idx, 3]
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
                  'joint': joint.astype(np.float), 'sequence': np.array(seq), 'mask': np.array(mask)}
        return sample


def _read_tripad_annotated(path, train):
    """
    Reads each annotated label/segment pair in the annotated TripAdvisor dataset
    :param path: the path to the data
    :param train: return the training set? Otherwise returns test set
    :return: iterator over iterators over label/segment pairs
    """
    with open(os.path.join(path, 'train.unique.json' if train else 'test.unique.json')) as f:
        for line in f:
            # get each line
            line_pars = json.loads(line)
            # zip all labels and segments of a singe comment
            yield zip(line_pars['segmentLabels'], line_pars['segments'])


def preprocess_tripad_annotated(path, train=True, include_tokens=True):
    """
    Builds a dataframe with aspect, polarity and corresponding segment
    for each comment in the annotated TripAdvisor dataset
    :param path: the path to the data
    :param train: return the training set? Otherwise returns test set
    :param include_tokens: include [CLS] and [SEP]?
    :return: pd.Dataframe: a dataframe with columns aspect, polarity and segment
    """
    data = []
    for obs in _read_tripad_annotated(path, train):
        for lab, seq in obs:
            if not lab:
                # non-related comment
                continue
            else:
                aspects = list(lab.keys())
                polarities = []
                # merge implicit polarities into explicit polarity categories
                for p in list(lab.values()):
                    if p in ['in', 'ix', 'ip']:
                        polarities.append(p[1])
                    elif p == 'il':
                        polarities.append('x')
                    elif p == 'i':
                        polarities.append('x')
                    else:
                        polarities.append(p)

                data.append({'aspects': aspects, 'polarities': polarities,
                             'joint': ['/'.join([a, b]) for a in aspects for b in polarities],
                             'segment': '[CLS] ' + seq + ' [SEP]' if include_tokens else seq})
    data = pd.DataFrame(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        help='Path to the annotated TripAdvisor dataset (should contain train/test.unique.json)',
                        required=True)
    args = parser.parse_args()
    data = TripadvisorDataset(args.data, BertTokenizer.from_pretrained('bert-base-uncased'))

    for i in range(len(data)):
        print(data[i]['sequence'])
