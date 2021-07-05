import random
from sys import stderr

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from cached_property import cached_property
from chainer.backends import cuda

from dqn_util import DoubleDQN


class ThreatDiscreteActionValue(chainerrl.action_value.DiscreteActionValue):
    def __init__(self, delta_threat, threat_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_threat = delta_threat
        self.threat_values = threat_values
        self.xp = cuda.get_array_module(threat_values)

    @cached_property
    def greedy_actions(self):
        min_threat_q_values = -1. * self.threat_values
        is_safe_action = self.threat_values < self.delta_threat
        checked_q_values = self.xp.where(is_safe_action, self.q_values.array, float(
            '-inf') * self.xp.ones_like(self.q_values.array))
        is_danger_state = self.xp.logical_not(np.any(
            is_safe_action, axis=1, keepdims=True))
        optimal_q_values = self.xp.where(
            is_danger_state, min_threat_q_values, checked_q_values)
        return chainer.Variable(
            optimal_q_values.argmax(axis=1).astype(np.int32))


class QFunction(chainer.Chain):
    def __init__(self, n_actions, n_hid1=50, n_hid2=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_hid1)
            self.l1 = L.Linear(None, n_hid2)
            self.l2 = L.Linear(None, n_actions)
        self.n_actions = n_actions

    def net(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return self.l2(h)

    def __call__(self, x, test=False):
        return chainerrl.action_value.DiscreteActionValue(self.net(x, test))


class RPQFunction(QFunction):
    def __init__(self, n_actions, threat_predictor,
                 delta_threat, n_hid1=50, n_hid2=50):
        super().__init__(n_actions, n_hid1, n_hid2)
        self.threat_predictor = threat_predictor
        self.delta_threat = delta_threat

    def __call__(self, x, test=False):
        h = self.net(x, test)

        if x.shape[0] != 1:
            return chainerrl.action_value.DiscreteActionValue(h)
        else:
            threat_values = self.threat_predictor.threat(x)[np.newaxis]

            self.safe_actions = np.where(threat_values < self.delta_threat)[1]

            return ThreatDiscreteActionValue(
                delta_threat=self.delta_threat,
                threat_values=threat_values,
                q_values=h)


class RPDQN(DoubleDQN):
    def act_and_train(self, obs, reward, greedy=False):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([obs], self.xp, self.phi))
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(
                    action_value.greedy_actions.data)[0]
        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        is_safe_future = len(self.q_function.safe_actions) > 0

        self.explorer.random_action_func = (lambda: random.choice(self.q_function.safe_actions)) \
            if is_safe_future else (lambda: greedy_action)
        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value) if not greedy else greedy_action

        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.action_parser(action)
