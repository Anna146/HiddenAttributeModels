import os
import sys
import tempfile

import numpy as np
import pescador as pescador
import torch
import torch.nn as nn
from torch import autograd

from .model_base import MODEL_BASE

sys.path.insert(0, "utils")
from model_statistics import *

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(33)

################################ MODEL ######################################################


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings - 1)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.sigm(self.fc(x))
        return out


class AttentionLayer2(nn.Module):
    def __init__(self, input_size, hidden_size_attention):
        super(AttentionLayer2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_attention)
        self.fc2 = nn.Linear(hidden_size_attention, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.sigm(self.fc1(x))
        out = self.sigm(self.fc2(x))
        return out


class Net(nn.Module):
    def __init__(self, hidden_size, char_len, utter_len, predicate_num, weights_matrix, hidden_size_attention=150, attention_type=0):
        super(Net, self).__init__()
        self.embedding, num_embeddings, self.embedding_len = create_emb_layer(weights_matrix, True)
        self.fc1 = nn.Linear(self.embedding_len, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.attention1 = AttentionLayer2(self.embedding_len, hidden_size_attention) if attention_type == 1 else AttentionLayer(self.embedding_len)
        self.attention2 = AttentionLayer2(self.embedding_len, hidden_size_attention) if attention_type == 1 else AttentionLayer(self.embedding_len)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.char_len = char_len
        self.utter_len = utter_len
        self.fc = nn.Linear(hidden_size, predicate_num)

    def forward(self, x):
        # 0 load embeddings
        x = self.embedding(x).view(-1, self.char_len, self.utter_len, self.embedding_len)

        # 1 combine words
        att_matrix = self.attention1(x)
        att_matrix = torch.where(x.narrow(3, 0, 1) != 0, att_matrix, torch.ones_like(att_matrix) * np.NINF)
        att_matrix = self.softmax2(att_matrix)
        att_matrix = torch.where(x.narrow(2, 0, 1).narrow(3, 0, 1) != 0, att_matrix, torch.zeros_like(att_matrix))
        weights = att_matrix.cpu().data.numpy()
        att_matrix = att_matrix.expand_as(x)
        x = torch.mul(att_matrix, x)
        x = torch.sum(x, dim=2)
        x = x.view(-1, self.char_len, self.embedding_len)

        # 2 combine utternaces
        att_matrix = self.attention2(x)
        att_matrix = torch.where(x.narrow(2, 0, 1) != 0, att_matrix, torch.ones_like(att_matrix) * np.NINF)
        att_matrix = self.softmax1(att_matrix)
        utter_weights = att_matrix.cpu().data.numpy()
        att_matrix = att_matrix.expand_as(x)
        x = torch.mul(att_matrix, x)
        x = torch.sum(x, dim=1)

        # 3 do feed forward
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc(out)
        out = torch.nn.functional.log_softmax(out, dim=1)

        return out, weights, utter_weights


########################################################################################################


@MODEL_BASE.register
class DoubleAtt(MODEL_BASE):
    @staticmethod
    def config():
        # model params
        hidden_size = 100
        hidden_size_attention = 150
        attention_type = 1  # 0 for no hid layer in attention, 1 for with hid layer

        # input params
        char_len = 100
        utter_len = 100
        embedding_len = 300
        predicate_num = 43
        dataset_len = 3000

        # optimizer params
        learning_rate = 0.001
        reg_lambda = 2e-7

        # training params
        num_epochs = 150
        batch_size = 32
        max_batch_epoch = dataset_len // batch_size

        model_path = ""
        model_name = "2attn"

        return locals().copy()  # ignored by sacred

    def __init__(self, *args, **kwargs):
        super(DoubleAtt, self).__init__(*args, **kwargs)
        self.p["model_path"] = "%s/%s_%s.pkl" % (self.p["models_dir"], self.p["model_name"], self.p["expname"])
        self.p["dump_dir"] = "%s/%s_%s.txt" % (self.p["dump_dir"], self.p["model_name"], self.p["expname"])
        self.p["results_file"] = "%s/%s_%s.txt" % (self.p["results_dir"], self.p["model_name"], self.p["expname"])
        self.weights = self.load_vocabulary()

    def build(self):
        p = self.p
        self.model = Net(
            hidden_size=p["hidden_size"],
            char_len=p["char_len"],
            utter_len=p["utter_len"],
            predicate_num=p["predicate_num"],
            weights_matrix=self.weights,
            hidden_size_attention=p["hidden_size_attention"],
            attention_type=p["attention_type"],
        )
        return self.model

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return self.model

    def load_vocabulary(self):
        weights = np.load(self.p["weights_path"])
        weights = np.append(weights.astype(float), np.zeros(shape=(1, self.p["embedding_len"])), axis=0)
        return weights

    def train(self, train_dir, kk=0, folds=10000, grid=False, grid_file=None):
        p = self.p
        net = self.build()
        net.to(device)
        net.train()

        # Loss and Optimizer
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=p["learning_rate"], weight_decay=p["reg_lambda"])

        # Pescador streams
        train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        streams_train = [pescador.Streamer(indexes_gen, ff, 1, p["utter_len"], p["char_len"], k=kk, mode="train", folds=folds) for ff in train_files]
        mux_stream_train = pescador.ShuffledMux(streams_train, random_state=33)

        word_idxs = np.empty(shape=(p["batch_size"], p["utter_len"] * p["char_len"]), dtype=int)
        labels = np.empty(shape=(p["batch_size"]), dtype=int)

        # Train the Model
        for epoch in range(p["num_epochs"]):
            print("Epoch " + str(epoch))
            for i, (word_idx, label, _, _) in enumerate(mux_stream_train):
                np.copyto(word_idxs[i % p["batch_size"]], word_idx)
                labels[i % p["batch_size"]] = label
                if i % p["batch_size"] == 0 and i != 0:
                    answers = autograd.Variable(torch.LongTensor(labels))
                    samples = torch.LongTensor(word_idxs)
                    answers = answers.to(device)
                    samples = samples.to(device)

                    optimizer.zero_grad()
                    outputs, weights, utter_weights = net(samples)
                    loss = criterion(outputs, answers)
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 20 == 0:
                        print("Epoch [%d/%d], Batch [%d], Loss: %.4f" % (epoch + 1, p["num_epochs"], i + 1, loss.item()))
                    if i // p["batch_size"] > p["max_batch_epoch"]:
                        break

            # Estimate intermediate
            if grid and (epoch + 1) % 10 == 0:
                net.eval()
                results_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
                streams_test = [
                    pescador.Streamer(indexes_gen, ff, 1, p["utter_len"], p["char_len"], k=kk, mode="test", folds=folds) for ff in train_files
                ]
                mux_stream_test = pescador.ChainMux(streams_test)
                for i, (word_idx, label, character, _) in enumerate(mux_stream_test):
                    samples = torch.LongTensor(word_idx.reshape((1, p["utter_len"] * p["char_len"])))
                    samples = samples.to(device)
                    output, _, _ = net(samples)
                    entry = output.cpu().data.numpy()[0]
                    results_file.write(
                        str(character[0])
                        + "\t"
                        + str(label[0])
                        + "\t"
                        + "\t".join([str(y) for y in sorted(enumerate(np.exp(entry)), key=lambda x: x[1], reverse=True)])
                        + "\n"
                    )
                results_file.close()

                mrr_character = compute_MRR_per_character(results_file.name)
                macro_mrr = compute_MRR_per_prof(results_file.name, 1)
                auroc = compute_auroc(results_file.name, 1)
                grid_file.write(str(epoch + 1) + "\t" + str(mrr_character) + "\t" + str(macro_mrr) + "\t" + str(auroc[0]) + "\n")
                grid_file.flush()
                os.remove(results_file.name)
                net.train()

        # Save the Model
        if grid == False:
            self.save(p["model_path"])

    def validate(self, test_file):
        p = self.p
        net = self.build()
        net = self.load(p["model_path"])
        net.to(device)
        net.eval()

        streamer = pescador.Streamer(indexes_gen, test_file, 1, p["utter_len"], p["char_len"])

        with open(p["results_file"], "w") as results_file:
            for word_idxs, labels, characters, sentences in streamer:
                samples = torch.LongTensor(word_idxs)
                samples = samples.to(device)

                output, weights, utter_weights = net(samples)

                i = 0
                for entry in output.cpu().data.numpy():
                    results_file.write(
                        str(characters[i])
                        + "\t"
                        + str(labels[i])
                        + "\t"
                        + "\t".join([str(y) for y in sorted(enumerate(np.exp(entry)), key=lambda x: x[1], reverse=True)])
                        + "\n"
                    )
                    i += 1

        compute_stats_dump(test_file=p["results_file"], dump_dir=p["dump_dir"])
