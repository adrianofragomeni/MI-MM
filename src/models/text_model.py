import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

class Sentence_Embedding(nn.Module):
    def __init__(
        self,
        path_resources,
        embd_dim=512,
        num_embeddings=66250,
        word_embedding_dim=300,
        max_words=16,
        output_dim=2048,
    ):
        super(Sentence_Embedding, self).__init__()
        self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.word_embd.weight.requires_grad=False
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(path_resources / "s3d_dict.npy")
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
            
        self.missingword_to_token = {}
        token_to_missingword = np.load(path_resources / "missing_words.npy")
        self.embedding_missingword=np.load(path_resources / "embeddings_missing_words.npy")
        for i, t in enumerate(token_to_missingword,num_embeddings):
            self.missingword_to_token[t] = i
            
    def update_embeddings(self):
        self.word_to_token.update(self.missingword_to_token)
        self.embedding_word=self.word_embd.weight.data.cpu().numpy()
        self.total_word_embd=np.vstack([self.embedding_word,self.embedding_missingword])

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
                
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x, device):
        x = self._words_to_ids(x)
        x = th.tensor(self.total_word_embd[x,:]).to(device)
        x = F.relu(self.fc1(x))
        x = th.max(x, dim=1)[0]
        return {'text_embedding': self.fc2(x),"text_features":x}
