import torch
import torch.nn as nn

import torch.nn.functional as F

import cogmen

log = cogmen.utils.get_logger()


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.args = args
        if args.use_highway:
            self.highway = Highway(size=input_dim, num_layers=1, f=F.relu)
            print("*******Using  Highway*******")
        else:
            self.highway = nn.Identity()
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.lin_7 = nn.Linear(hidden_size, 7)
        self.linear = nn.Linear(input_dim, tag_size)
        if args.class_weight:
            if args.dataset == "iemocap":
                self.loss_weights = torch.tensor(
                    [
                        1 / 0.086747,
                        1 / 0.144406,
                        1 / 0.227883,
                        1 / 0.160585,
                        1 / 0.127711,
                        1 / 0.252668,
                    ]
                ).to(args.device)
            elif args.dataset == "iemocap_4":
                self.loss_weights = torch.tensor(
                    [
                        1 / 0.1426370239929562,
                        1 / 0.2386088487783403,
                        1 / 0.37596302003081666,
                        1 / 0.24279110719788685,
                    ]
                ).to(args.device)

            elif args.dataset == "mosei":
                if args.emotion == "happiness":
                    self.loss_weights = torch.tensor(
                        [1 / 0.4717985331342896, 1 / 0.5282014668657103]
                    ).to(args.device)
                elif args.emotion == "anger":
                    self.loss_weights = torch.tensor(
                        [1 / 0.7796456156292594, 1 / 0.22035438437074056]
                    ).to(args.device)
                elif args.emotion == "disgust":
                    self.loss_weights = torch.tensor(
                        [1 / 0.8083987797754267, 1 / 0.19160122022457324]
                    ).to(args.device)
                elif args.emotion == "fear":
                    self.loss_weights = torch.tensor(
                        [1 / 0.8083987797754267, 1 / 0.19160122022457324]
                    ).to(args.device)
                elif args.emotion == "surprise":
                    self.loss_weights = torch.tensor(
                        [1 / 0.9220484195495554, 1 / 0.0779515804504446]
                    ).to(args.device)
                elif args.emotion == "sadness":
                    self.loss_weights = torch.tensor(
                        [1 / 0.7288894658272214, 1 / 0.2711105341727786]
                    ).to(args.device)
                elif args.emotion == "2class":
                    self.loss_weights = torch.tensor(
                        [3.445241612154463, 1.4089575422851226]
                    ).to(args.device)
                elif args.emotion == "7class":
                    self.loss_weights = torch.tensor(
                        [
                            26.63458401305057,
                            10.307449494949495,
                            6.422895357985838,
                            4.614754098360656,
                            3.0869729627528835,
                            7.237145390070922,
                            32.33069306930693,
                        ]
                    ).to(args.device)
                elif args.emotion == "multilabel":
                    self.loss_weights = torch.tensor(
                        [
                            1 / 0.5356517935258093,
                            1 / 0.2588801399825022,
                            1 / 0.2158792650918635,
                            1 / 0.08276465441819772,
                            1 / 0.1767716535433071,
                            1 / 0.1,
                        ]
                    ).to(args.device)
                    self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
                else:
                    self.loss_weights = torch.tensor(
                        [1 / 0.303032097595, 1 / 0.696967902404]
                    ).to(args.device)

            self.nll_loss = nn.NLLLoss(self.loss_weights)
            print("*******weighted loss*******")
        else:
            self.nll_loss = nn.NLLLoss()
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def get_prob(self, h, text_len_tensor):
        if self.args.use_highway:
            h = self.highway(h)
        hidden = self.drop(F.relu(self.lin1(h)))
        if self.args.emotion == "7class":
            scores = self.lin_7(hidden)
        else:
            scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=1)
        return log_prob

    def forward(self, h, text_len_tensor):
        if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
            if self.args.use_highway:
                h = self.highway(h)
            hidden = self.drop(F.relu(self.lin1(h)))
            scores = self.lin2(hidden)
            # y_hat = torch.sigmoid(scores) > 0.5
            y_hat = scores > 0
            return y_hat

        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
            if self.args.use_highway:
                h = self.highway(h)
            hidden = self.drop(F.relu(self.lin1(h)))
            scores = self.lin2(hidden)
            loss = self.bce_loss(scores, label_tensor.float())
            if self.args.class_weight:
                loss = (loss * self.loss_weights).mean()
            # breakpoint()
            return loss

        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)
        return loss


class MaskedEmotionAtt(nn.Module):
    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s : s + cur_len]
            z = h[s : s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = z.unsqueeze(0) * probs.unsqueeze(-1)
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s : s + cur_len, :] = out
            s += cur_len

        return ret


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x
