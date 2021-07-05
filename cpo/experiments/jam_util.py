import cmath
import itertools
from time import sleep

import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

LIDAR_STEP = 1


def plot(c, char='.', color=None):
    if color is None:
        plt.plot(c.real, c.imag, char)
    else:
        plt.plot(c.real, c.imag, char, color=color)


def rot(om):
    return np.exp(1.j * om)


class Field:
    def __init__(self, height=10, width=10, exit_=1, space=0.02):
        assert exit_ < width
        self.height = height
        self.width = width
        self.exit = exit_
        self.wall = np.array([])
        self.space = space
        self.xline(0, width, y=0)
        self.xline(0, width - self.exit, y=height)
        self.yline(0, height, x=0)
        self.yline(0, height, x=width)
        self.mark_point = (width - self.exit / 2) + height * 1.j

    def xline(self, xmin, xmax, y):
        x_ = np.arange(xmin, xmax, self.space)
        y_ = np.ones(len(x_)) * y
        self.wall = np.append(self.wall, x_ + y_ * 1.j)

    def yline(self, ymin, ymax, x):
        y_ = np.arange(ymin, ymax, self.space)
        x_ = np.ones(len(y_)) * x
        self.wall = np.append(self.wall, x_ + y_ * 1.j)

    def draw(self, color='green', lw=3):
        x = [self.width - self.exit, 0, 0, self.width, self.width]
        y = [self.height, self.height, 0, 0, self.height]
        plt.plot(x, y, lw=3, color=color)


class Agent:
    def __init__(self, r=0.1, pos=0.j, front=1 + 0.j, v=0):
        self.r = r
        self.pos = pos
        self.front = front
        self.v = v

    def borders(self):
        arr = np.arange(-np.pi, np.pi, np.pi / 18)  # per 10 deg
        arr = rot(arr) * self.r
        return self.pos + self.front * arr

    def draw(self):
        raise NotImplementedError

    def draw_edge(self, color):
        ax = plt.axes()
        c = pch.Circle(xy=(self.pos.real, self.pos.imag),
                       radius=self.r, ec=color, fill=False, lw=3)
        ax.add_patch(c)
        ax.set_aspect('equal')

    def tick(self):
        # select action and call update
        raise NotImplementedError

    def update(self, acc, omega):
        self.pos += self.front * self.v
        self.v = np.clip(self.v + acc, 0, 0.06)
        self.front *= rot(omega)


class MyCar(Agent):
    def __init__(self, pos=0.2 + 0.2j, front=1 + 0.j, v=0):
        super().__init__(pos=pos, front=front, v=v)

        self.acc_list = [-0.02, 0, 0.02]
        self.omega_list = [-0.3, -0.1, 0, 0.1, 0.3]
        #self.omega_list = [-0.15, -0.05, 0, 0.05, 0.15]
        self.action_list = list(itertools.product(
            self.acc_list, self.omega_list))

    def draw(self):
        self.draw_edge('blue')

        if abs(self.v) < 1e-4:
            v = 0.05 * self.front
            plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag, angles='xy',
                       scale_units='xy', scale=0.2, color='orange')
        else:
            if self.v < 0:
                v = self.v * self.front
                plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag, angles='xy',
                           scale_units='xy', scale=0.2, color='red')
            else:
                v = self.v * self.front
                plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag, angles='xy',
                           scale_units='xy', scale=0.2, color='black')

    def tick(self, action):
        acc, omega = action
        self.update(acc, omega)

    def update(self, acc, omega):
        self.pos += self.front * self.v
        self.v = np.clip(self.v + acc, -0.1, 0.1)
        self.front *= rot(omega)


class OtherCar(Agent):
    def __init__(self, pos=0.2 + 0.2j, front=1 + 0.j, v=0):
        super().__init__(pos=pos, front=front, v=v)

        self.acc_list = [-0.02, 0, 0.02]
        self.acc_p = [0.2, 0.6, 0.2]
        self.omega_list = [-0.15, -0.05, 0, 0.05, 0.15]
        self.action_list = list(itertools.product(
            self.acc_list, self.omega_list))

    def draw(self):
        self.draw_edge('red')

        if self.v < 1e-4:
            v = 0.05 * self.front
            plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag, angles='xy',
                       scale_units='xy', scale=0.2, color='orange')
        else:
            v = self.v * self.front
            plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag, angles='xy',
                       scale_units='xy', scale=0.2, color='black')

    def tick(self):
        acc = np.random.choice(self.acc_list, p=self.acc_p)
        om = np.random.choice(self.omega_list)
        self.update(acc, om)


class Environment(Env):
    def __init__(self, n_others, field, view_rad=1, initial_pos=0.2 + 0.2j, render=False,
                 file='crash.log', all_log=False, lmd=30, max_step=None, cur=False):
        self.my = MyCar(pos=initial_pos)
        self.others = np.array([OtherCar() for _ in range(n_others)])
        self.field = field
        self.view_rad = view_rad
        self.episode_cnt = 0
        self.step_cnt = 0
        self.crash_cnt = 0
        self.total_step = 0
        self.episode_reward = 0
        self.first = True
        self.locate_all()
        self.update_rpos()
        self.prev_frame = np.zeros(360)
        self.prev = self.progress()
        self.flag_render = render
        self.file = 'results/' + file
        self.all_log = all_log
        self.result_agent = None
        self.lmd = lmd
        self.max_step = max_step
        self.cur = cur
        self.reward_range = [-1000, 200]
        self.metadata = None
        self.now_checking = False
        self.crash_checking = False
        self.next_check = 20000

    @property
    def observation_space(self):
        lim = np.ones(396) * 10
        return Box(low=-lim, high=lim)

    @property
    def action_space(self):
        lim = np.array([0.02, 0.3])
        return Box(low=-lim, high=lim)

    def print_agent_info(self, agent):
        print('pos:', agent.pos, ' v:', agent.v,
              ' th:', np.degrees(np.angle(agent.front)))

    def info(self):
        self.print_agent_info(self.my)
        for agent in self.others:
            self.print_agent_info(agent)

    def progress(self):
        return -np.round(abs(self.field.mark_point - self.my.pos) * 20)

    def update_rpos(self):
        self.rpos = self.field.wall
        self.wall_rpos = (self.field.wall - self.my.pos) / self.my.front
        self.wall_rpos = self.wall_rpos[abs(self.wall_rpos) < self.view_rad]

        for agent in self.others:
            self.rpos = np.append(self.rpos, agent.borders())

        self.rpos = (self.rpos - self.my.pos) / self.my.front

    def near_agents(self, n=8):
        arr = abs(np.array([agent.pos - self.my.pos for agent in self.others]))
        x = np.argsort(arr)[:n]
        return self.others[x]

    def render(self):
        plt.clf()
        plt.axis('off')
        self.my.draw()
        for agent in self.others:
            agent.draw()
        self.field.draw()

        plt.pause(1e-6)

    def view(self):
        return self.rpos[abs(self.rpos) < self.view_rad]

    # only wall
    def LIDAR(self, step=LIDAR_STEP):
        view = self.wall_rpos
        if len(view) is 0:
            return np.zeros(int(360 / LIDAR_STEP))
        phase = np.vectorize(cmath.phase)(view)
        obs = []
        for i in np.arange(-180, 180, step):
            hit = view[(np.radians(i) < phase) & (
                phase < np.radians(i + step))]
            obs.append(abs(hit).min() if len(hit) > 0 else 0)
        return np.array(obs)

    def obs(self):
        near = self.near_agents()
        pos = (np.array([agent.pos for agent in near]) -
               self.my.pos) / self.my.front
        pos = np.c_[pos.real, pos.imag].T
        angle = np.angle(np.array([agent.front for agent in near]) / self.my.front)[
            np.newaxis]
        v = np.array([agent.v for agent in near])[np.newaxis]

        data = np.concatenate((pos, angle, v), axis=0).T

        return np.append(np.append(self.LIDAR(), data.ravel()),
                         [self.my.pos.real, self.my.pos.imag,
                          np.angle(self.my.front), self.my.v])

    def check_crash_wall(self, agent):
        return np.any(abs(self.wall_rpos) < self.my.r)

    def check_out_area(self, agent):
        x, y, r, w, h = agent.pos.real, agent.pos.imag, agent.r, self.field.width, self.field.height
        return not(r < x < w - r) or not(r < y < h - r) \
            or (abs(agent.pos.real) < 0.5 and abs(agent.pos.imag) < 0.5) \
            or (abs((agent.pos - self.field.width).real) < 0.5 and abs((agent.pos - self.field.width).imag) < 0.5) \
            or (abs((agent.pos - 1.j * self.field.height).real) < 0.5 and abs((agent.pos - 1.j * self.field.height).imag) < 0.5)

    def check_crash(self):
        if self.check_crash_wall(self.my):
            self.crash_cnt += 1
            return True

        for agent in self.others:
            if abs(agent.pos - self.my.pos) < agent.r + self.my.r:
                self.crash_cnt += 1
                return True
        return False

    def goal(self):
        return self.my.pos.imag > self.field.height

    def finished(self):
        return self.crashed or self.goal()

    def reward(self):
        now = self.progress()
        re = now - self.prev
        self.prev = now

        if self.goal():
            re += 10
        if self.crashed:
            re -= self.now_lmd
        if abs(self.my.v) < 0.005:
            re -= 0.05
        return re - 0.05

    # call after my pos is decided
    def locate(self, agent):
        r = agent.r
        pos = self.my.pos
        while abs(self.my.pos - pos) < self.view_rad:
            x = np.random.uniform(r, self.field.width - r)
            y = np.random.uniform(r, self.field.height - r)
            pos = x + 1.j * y
        agent.pos = pos
        agent.front = np.exp(1.j * np.random.uniform(-np.pi, np.pi))
        agent.v = np.random.randint(4) * 0.02

    def locate_all(self):
        for agent in self.others:
            self.locate(agent)

    def reset(self):
        self.total_step += self.step_cnt

        if self.episode_cnt is not 0:
            if self.all_log or self.episode_cnt % 20 is 0:
                with open(self.file, mode='a') as f:
                    f.write('Episode ' + str(self.episode_cnt))
                    f.write(' step: ' + str(self.step_cnt))
                    f.write(' total_reward: ' + str(round(
                        self.episode_reward, 1)))
                    f.write(' total_step: ' + str(self.total_step))
                    f.write(' crashed: ' + str(int(self.crashed)))
                    f.write(' crash_cnt: ' + str(self.crash_cnt) + '\n')
                    f.close()

                print('Episode', self.episode_cnt, end=' ')
                print('step:', self.step_cnt, end=' ')
                print('total_reward:', round(self.episode_reward, 1), end=' ')
                print('total_step:', self.total_step, end=' ')
                print('crashed:', int(self.crashed), end=' ')
                print('crash_cnt:', self.crash_cnt)

        if self.crash_checking:
            with open('results/crash.txt', mode='a') as f:
                f.write('Step: ' + str(self.total_step))
                f.write(' Crashed: ' + str(int(self.crashed)) + '\n')
                f.close()
                self.crash_checking = False

        if self.now_checking:
            with open('results/reward.txt', mode='a') as f:
                f.write('Step: ' + str(self.total_step))
                f.write(' Reward: ' + str(round(self.episode_reward, 1)) + '\n')
                f.close()
                self.next_check += 20000
                self.now_checking = False
                self.crash_checking = True

        if self.total_step > self.next_check:
            self.now_checking = True

        self.step_cnt = 0
        self.episode_cnt += 1

        self.my = MyCar()
        self.locate_all()

        candi = (self.lmd - 5) / self.max_step * self.total_step + 5
        self.now_lmd = candi if self.cur else self.lmd

        self.prev = self.progress()
        self.episode_reward = 0
        self.update_rpos()
        return self.obs()

    def step(self, action):
        self.step_cnt += 1
        self.my.tick(action)
        for agent in self.others:
            agent.tick()
            if self.check_out_area(agent):
                self.locate(agent)

        self.update_rpos()
        info = {}
        self.crashed = self.check_crash()
        r = self.reward()
        self.episode_reward += r
        if self.flag_render:
            self.render()
            if self.finished():
                sleep(5)

        info['crashed'] = int(self.crashed)
        return Step(self.obs(), r, self.finished(), **info)
