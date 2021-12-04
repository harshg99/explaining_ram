import torch.nn as nn

import modules


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1]

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes,
    ):
        """
        """
        super().__init__()

        self.std = std

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.Critic(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)

        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi
