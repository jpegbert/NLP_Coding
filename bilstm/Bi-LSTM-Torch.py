import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


"""
BiLSTM的pytorch实现
"""


dtype = torch.FloatTensor

sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)

word2idx = {w: i for i, w in enumerate(list(set(sentence.split())))}
idx2word = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word2idx) # classification problem
max_len = len(sentence.split())
n_hidden = 5


def make_data(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split() # ['Github', 'Actions', 'makes', ...]
    for i in range(max_len - 1):
        input = [word2idx[n] for n in words[:(i + 1)]] # input = [18 7 3]
        input = input + [0] * (max_len - len(input)) # input = [18 7 3 0 'it', ..., 0]
        target = word2idx[words[i + 1]] # target = [0]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


# input_batch: [max_len - 1, max_len, n_class]
input_batch, target_batch = make_data(sentence)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, 16, True)


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1 * 2, batch_size, n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size, n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state)) # [max_len, batch_size, n_hidden * 2]
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model


model = BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10000):
    for x, y in loader:
      pred = model(x)
      loss = criterion(pred, y)
      if (epoch + 1) % 1000 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

# Pred
predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([idx2word[n.item()] for n in predict.squeeze()])

