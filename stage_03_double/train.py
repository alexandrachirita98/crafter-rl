"""Stage 03 -- Double DQN (lab notebook section 5, TASK 4).

van Hasselt, Guez & Silver 2016, https://arxiv.org/abs/1509.06461

Stage 02's max over 17 noisy estimates is biased upward: the luckiest estimate
is the one picked, so its error never averages away. Double DQN splits the two
jobs -- the online net picks the next action, the target net prices it:

    a*     = argmax_a' Q_online(s', a')
    target = r + gamma * mask * Q_target(s', a*)

Everything else is imported from stage 02. Expect mean-Q lower than stage 02's
at the same step, return equal or better.

Run:
    python stage_03_double/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import DQN, main as stage_02_main

STAGE_ID = 'stage_03_double'


class DoubleDQN(DQN):

    def _update(self, states, actions, rewards, states_, mask):
        # Both lines stay under no_grad: letting the argmax carry gradients
        # backprops through the target and destabilises training silently.
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)      # the stage-02 line was .max()

        target_qsa = rewards + self._gamma * qsa_ * mask

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


def main(argv=None):
    return stage_02_main(argv, agent_cls=DoubleDQN, stage_id=STAGE_ID)


if __name__ == '__main__':
    main()
