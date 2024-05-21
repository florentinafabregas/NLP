import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
import pickle
import sys

# Load the dictionaries
with open('models/bilstm_word/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)
with open('models/bilstm_word/id2tag.pkl', 'rb') as f:
    id2tag = pickle.load(f)
with open('models/bilstm_word/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('models/bilstm_word/nertags.pkl', 'rb') as f:
    nertags = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, total_words, num_class):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.wordembed = nn.Embedding(total_words, embedding_size)
        self.dropout = nn.Dropout(p=0.5)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, num_class)

    def forward(self, x, xlengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        word_embedding = self.wordembed(x)
        word_embedding = self.dropout(word_embedding)

        out, (h, c) = self.bilstm(word_embedding)
        out = self.linear(out)
        out = out.view(-1, out.shape[2])
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out

# Initialize your model instance with the same architecture as the trained model
model = BiLSTM(embedding_size=100, hidden_size=100, total_words=len(vocab), num_class=len(nertags))

# Load the saved model state dictionary
model.load_state_dict(torch.load('models/bilstm_word/trained_bilstm_model_state_dict.pth'))

# Put the model in evaluation mode
model.eval()

def out_predictions(model, loader, output_file):
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for step, (X, Y, xlen) in enumerate(loader):
                Y = pack_padded_sequence(Y, xlen, batch_first=True, enforce_sorted=False)
                Y, _ = pad_packed_sequence(Y, batch_first=True)
                ypred = model(X.long().to(device), xlen.to(device))
                ypred = torch.argmax(ypred.to('cpu'), dim=1)
                ypred = ypred.view(Y.shape[0], -1)
                for i in range(len(ypred)):
                    for j in range(len(ypred[i])):
                        word = id2word[int(X[i, j])]
                        tag = id2tag[int(ypred[i, j])]
                        f.write(f"{word}\t{tag}\n")
                    f.write('\n')

def load_data(datapath):
    sentences = []
    tags = []
    with open(datapath) as f:
        lines = f.readlines()
        sentence = []
        tag = []
        for line in lines:
            line = line.strip()  
            if line: 
                word, tag_label = line.split('\t')
                if vocab is not None:
                    if word in vocab.keys():
                        sentence.append(vocab[word])
                    else:
                        sentence.append(vocab['<oov>'])
                if nertags is not None:
                    tag.append(nertags[tag_label])
            else:  
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []

    max_length = max(len(x) for x in sentences)
    x_lengths = [len(x) for x in sentences]
    X_test = []
    Y_test = []
    for sent, tag in zip(sentences, tags):
        length_to_append = max_length - len(sent)
        X_test.append(sent + [0] * length_to_append)  
        Y_test.append(tag + [0] * length_to_append) 

    X_test = torch.Tensor(X_test)
    Y_test = torch.Tensor(Y_test)
    x_lengths = torch.Tensor(x_lengths)

    return X_test, Y_test, x_lengths

# SPAN-F1 SCORE
def readBIO(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[1])
    return ents

def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
            #print(end-beg)
    return spans

def getInstanceScores(predPath, goldPath):
    goldEnts = readBIO(goldPath)
    predEnts = readBIO(predPath)
    entScores = []
    tp = 0
    fp = 0
    fn = 0
    for goldEnt, predEnt in zip(goldEnts, predEnts):
        goldSpans = toSpans(goldEnt)
        predSpans = toSpans(predEnt)
        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap
        
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide noise_type and rate')
        
    else:
        noise_type = sys.argv[1]
        rate = sys.argv[2]

        testdatapath = f'data/altered/{noise_type}_rate_{rate}.txt'

        Xtest, Ytest, x_testlengths = load_data(testdatapath)

        testdataset = TensorDataset(Xtest, Ytest, x_testlengths)
        loader_test = DataLoader(testdataset, batch_size=1, shuffle=False)
        prediction_path = f'predictions/bilstm_word/{noise_type}_rate_{rate}.txt'

        # Output predictions
        out_predictions(model, loader_test, prediction_path)
        span_f1_score = getInstanceScores(prediction_path,'data/gold.txt')
        print('Span-F1 score: ', span_f1_score)
