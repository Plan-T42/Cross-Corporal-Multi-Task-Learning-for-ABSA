import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Embeddings:
    def __init__(self, model='bert-base-uncased', max_sequence_length=128):
        self.sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.bert = BertModel.from_pretrained(model).to(self.device)
        self.bert.eval()

    def _tokenize(self, sequences):
        """
        Tokenizes the input sequences, should begin with [CLS] and end with [SEP]
        :param sequences: iterable of input sequences
        :return: list: tokenized sequences
        """
        tokenized = []
        for seq in sequences:
            tokenized.append(self.tokenizer.tokenize(seq))
        return tokenized

    def _get_vocab_ids(self, tokens):
        """
        Turns an iterable of lists of tokens into WordPiece embeddings
        :param tokens: a iterable of lists of tokens
        :return: list: list of lists of token ids
        """
        ids = []
        for tok in tokens:
            ids.append(self.tokenizer.convert_tokens_to_ids(tok))
        return ids

    def get_embeddings(self, sequences, all=False):
        """
        Gets BERT embeddings for an iterable of token id lists
        :param token_ids: batch of sequences in token ids
        :param all: should BERT return all or just the last hidden layers?
        :return: tensor: batch of sequence embeddings
        """
        tokens = self._tokenize(sequences)
        token_ids = self._get_vocab_ids(tokens)
        embeddings = []
        for seq in token_ids:
            seq = torch.tensor([seq]).to(self.device)
            with torch.no_grad():
                embedding, _ = self.bert(seq, output_all_encoded_layers=all)
            embeddings.append(embedding.cpu().squeeze())
        return embeddings


if __name__ == '__main__':
    emb = Embeddings()
    embed = emb.get_embeddings(['[CLS] This is a test [SEP]', '[CLS] Another test [SEP]'], all=True)
    print(embed)
