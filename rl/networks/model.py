import torch
import torch.nn as nn


from rl.networks.distributions import Bernoulli, Categorical, DiagGaussian, DiagGaussianEqui
from .srnn_model import SRNN
from .selfAttn_srnn_temp_node import selfAttn_merge_SRNN
from .equi_rnn import equi_SRNN

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        use_equi=False
        if base == 'srnn':
            base=SRNN
        elif base == 'selfAttn_merge_srnn':
            base = selfAttn_merge_SRNN
        elif base == 'equi_srnn':
            #("EQIOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            base = equi_SRNN
            use_equi = True
        else:
            raise NotImplementedError

        self.base = base(obs_shape, base_kwargs)
        self.srnn = True

        if action_space.__class__.__name__ == "Discrete":
            if use_equi:
                print("NUM OUTPUTS ", action_space.n)
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            if use_equi:
                #print("USING EQUI DIAG GAUSSIAN")
                self.dist = DiagGaussianEqui()
            else:
                num_outputs = action_space.shape[0]
                #print("BOX NUM OUTPUTS ", num_outputs)
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        #for key in inputs:
            #print("INPUT KEY: ", key, inputs[key].shape)
        #for key in rnn_hxs:
            #print("RNN HXS KEY: ", key, rnn_hxs[key].shape)
        #print("MASK SHAPE", masks.shape)

        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            #print("SRNN", self.base)
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)

        else:
            #print("WHAT IS tHIS", self.base)
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            #print("DETERMINISTIC")
            action = dist.mode()
        else:
            #print("GAUSSIAN")
            action = dist.sample()
            #print("ACTION ", action.shape, action)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



