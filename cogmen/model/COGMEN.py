import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .GNN import GNN
from .Classifier import Classifier
from .functions import batch_graphify
import cogmen

log = cogmen.utils.get_logger()


class COGMEN(nn.Module):
    def __init__(self, args):
        super(COGMEN, self).__init__()
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.concat_gin_gout = args.concat_gin_gout

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.gcn = GNN(g_dim, h1_dim, h2_dim, args)
        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        # [batch_size, mx_len, D_g]
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )

        graph_out = self.gcn(features, edge_index, edge_type)
        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )

        return loss
