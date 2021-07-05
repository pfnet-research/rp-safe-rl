import cmath
import itertools
import os
from time import sleep, time

import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.patches as patches
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
from gym import Env


def rot_deg(deg):
    return cmath.rect(1, np.radians(deg))


def rot(rad):
    return cmath.rect(1, rad)


LIDAR_STEP = 1

_arr = np.arange(-180, 180, LIDAR_STEP) + LIDAR_STEP / 2
_arr = np.exp(np.radians(_arr) * 1j)


def invLIDAR(obs, xp, step=LIDAR_STEP):
    pred = (1e-6 < obs) & (obs < 10)
    return (obs * _arr)[pred]


class Circuit:
    def __init__(self, step, curve_step, width):
        self.pos = 0.j
        self.front = 1 + 0.j
        self.width = width
        self.lwall = 1j * self.width
        self.rwall = -1j * self.width

        self.cpos = [self.pos]
        self.wall = [self.lwall, self.rwall]
        self.left = [self.lwall]
        self.right = [self.rwall]
        self.front_arr = [self.front]

        self.step = step
        self.curve_step = curve_step

    def record(self):
        self.cpos.append(self.pos)
        self.lwall = self.pos + self.width * self.front * rot_deg(90)
        self.rwall = self.pos + self.width * self.front * rot_deg(-90)
        self.wall.append(self.lwall)
        self.wall.append(self.rwall)

        self.left.append(self.lwall)
        self.right.append(self.rwall)

        self.front_arr.append(self.front)

    def convert_to_ndarray(self):
        self.wall = np.array(self.wall)
        self.cpos = np.array(self.cpos)
        self.left = np.array(self.left)
        self.right = np.array(self.right)

    def straight(self, l):
        for _ in range(int(l / self.step)):
            self.pos += self.step * self.front
            self.record()

    def curve(self, r, deg):
        if deg > 0:
            cent = self.pos + self.front * rot_deg(90) * r
            step_theta = self.curve_step
        else:
            cent = self.pos + self.front * rot_deg(-90) * r
            step_theta = -self.curve_step

        vec = self.pos - cent

        for _ in range(int(deg / step_theta)):
            vec *= rot_deg(step_theta)
            self.front *= rot_deg(step_theta)
            self.pos = cent + vec
            self.record()

    def turn(self, deg):
        self.curve(self.width, deg)

    def go_home(self):
        th = cmath.phase(self.front)
        if th > 0:
            limit = -self.width * (1 - np.cos(th))
        else:
            limit = self.width * (1 - np.cos(th))

        while abs(self.pos.imag) > limit:
            self.straight(self.step)

        self.turn(np.degrees(-th))

        while self.pos.real < 0:
            self.straight(self.step)

    def draw(self):
        color = 'royalblue'
        plt.plot(self.left.real, self.left.imag, lw=3, color=color)
        plt.plot(self.right.real, self.right.imag, lw=3, color=color)


class Agent:
    def __init__(self, r=0.1, pos=0.j, front=1 + 0.j, v=0):
        self.r = r
        self.pos = pos
        self.front = front
        self.v = v
        self.acc_list = [-0.02, 0, 0.02]
        self.omega_list = [-0.15, -0.05, 0, 0.05, 0.15]
        self.action_list = list(itertools.product(
            self.acc_list, self.omega_list))

    def draw_border(self, color='green'):
        ax = plt.axes()
        c = pch.Circle(xy=(self.pos.real, self.pos.imag),
                       radius=self.r, ec=color, fill=False, lw=3)
        ax.add_patch(c)
        ax.set_aspect('equal')

    def draw(self):
        plt.plot(self.pos.real, self.pos.imag, '*')
        v = self.v * self.front
        if abs(v) < 0.005:
            v = 0.05 * self.front
            plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag,
                       angles='xy', scale_units='xy', scale=0.2, color='orange')
        else:
            if self.v < 0:
                plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag,
                           angles='xy', scale_units='xy', scale=0.2, color='red')
            else:
                plt.quiver(self.pos.real, self.pos.imag, v.real, v.imag,
                           angles='xy', scale_units='xy', scale=0.2, color='black')

        self.draw_border()

    def tick(self, action_num):
        acc, omega = self.action_list[action_num]

        self.pos += self.front * self.v
        self.v += acc
        self.v = np.clip(self.v, -0.04, 0.1)

        self.front *= rot(omega)


class Environment(Env):
    def __init__(self, circuit, lmd=200, random_init=False,
                 result_dir='results', file='crash.log', all_log=False, render=False):
        self.agent = Agent()
        self.circuit = circuit
        self.obj = circuit.wall
        self.prev = self.progress()
        self.episode_cnt = 0
        self.step_cnt = 0
        self.crash_cnt = 0
        self.total_step = 0
        self.episode_reward = 0
        self.first = True
        self.crash_cnt = 0
        self.random_init = random_init
        self.cnt = 0
        self.sum = 0
        self.result_dir = result_dir
        self.file = os.path.join(self.result_dir, file)
        self.all_log = all_log
        self.lmd = lmd
        self.optimizer = None
        self.next_save_cnt = 1
        self.result_agent = None
        self.flag_render = render

    def update_rpos(self):
        self.rpos = (self.obj - self.agent.pos) / self.agent.front

    def progress(self):
        return abs(self.circuit.cpos - self.agent.pos).argmin()

    def dir_diff(self):
        now = self.progress()
        l = len(self.circuit.cpos)
        return cmath.phase(
            (self.circuit.cpos[(now + 1) % l] - self.circuit.cpos[now]) / self.agent.front)

    def render(self):
        plt.clf()
        plt.axis('off')
        self.circuit.draw()
        self.agent.draw()
        plt.pause(1e-6)

    def view(self, rad=1):
        return self.rpos[abs(self.rpos) < rad]

    def LIDAR(self, step=LIDAR_STEP):
        view = self.view(rad=1)

        phase = np.vectorize(cmath.phase)(view)
        obs = []
        for i in np.arange(-180, 180, step):
            hit = view[(np.radians(i) < phase) & (
                phase < np.radians(i + step))]
            obs.append(abs(hit).min() if len(hit) > 0 else 0)
        return np.array(obs)

    def obs(self):
        return np.append(self.LIDAR(), [self.agent.pos.real, self.agent.pos.imag, self.dir_diff(
        ), self.agent.v]).astype(np.float32)

    def check_crash(self):
        res = np.any(abs(self.rpos) < self.agent.r)
        if res:
            self.crash_cnt += 1
            with open(os.path.join(self.result_dir, 'crash_pos.log'), mode='a') as f:
                f.write(str(self.total_step) + ' ')
                f.write(str(self.agent.pos.real) + ' ')
                f.write(str(self.agent.pos.imag) + '\n')
                f.close()
        return res

    def finished(self):
        return self.crashed

    def reward(self):
        re = 0
        now = self.progress()
        if abs(now - self.prev) < len(self.circuit.cpos) // 4:
            re += (now - self.prev)

        self.prev = now
        if abs(self.agent.v) < 0.005:
            re -= 1
        if self.crashed:
            re -= self.lmd

        return re

    def reset(self):
        self.total_step += self.step_cnt
        if self.total_step >= self.next_save_cnt * 10 ** 5:
            self.result_agent.save(os.path.join(
                self.result_dir, 'agent' + str(self.next_save_cnt)))
            self.next_save_cnt += 1

        if self.episode_cnt is not 0:
            if self.all_log or self.episode_cnt % 10 is 0:
                with open(self.file, mode='a') as f:
                    f.write('Episode ' + str(self.episode_cnt))
                    f.write(' step: ' + str(self.step_cnt))
                    f.write(' total_reward: ' + str(round(
                        self.episode_reward, 1)))
                    f.write(' total_step: ' + str(self.total_step))
                    f.write(' crash_cnt: ' + str(self.crash_cnt) + '\n')
                    f.close()

                print('Episode', self.episode_cnt, end=' ')
                print('step:', self.step_cnt, end=' ')
                print('total_reward:', round(self.episode_reward, 1), end=' ')
                print('total_step:', self.total_step, end=' ')
                print('crash_cnt:', self.crash_cnt)

        self.step_cnt = 0
        self.episode_cnt += 1
        if self.random_init:
            sz = len(self.circuit.cpos)
            n = np.random.randint(sz)
            pos = self.circuit.cpos[n]
            rad = np.pi / 4 * (2 * np.random.rand() - 1)
            front = self.circuit.front_arr[n] * rot(rad)
            self.agent = Agent(pos=pos, front=front, v=0)

        else:
            self.agent = Agent()

        self.prev = self.progress()
        self.episode_reward = 0
        self.update_rpos()
        return self.obs()

    def step(self, action_num):
        self.step_cnt += 1
        self.agent.tick(action_num)
        self.update_rpos()

        obs = self.obs()
        self.crashed = self.check_crash()

        r = self.reward()
        self.episode_reward += r

        if self.flag_render:
            self.render()

            if self.finished():
                sleep(3)

        return obs, r, self.finished(), {}


def default_circuit():
    circuit = Circuit(step=0.02, curve_step=1, width=0.3)

    circuit.straight(2)
    circuit.turn(45)
    circuit.straight(1)
    circuit.curve(1, 225)
    circuit.curve(0.5, -145)
    circuit.straight(1.5)
    circuit.curve(1.2, 75)
    circuit.straight(0.5)
    circuit.turn(110)
    circuit.straight(0.5)
    circuit.turn(-90)
    circuit.straight(0.5)
    circuit.curve(0.5, 120)
    circuit.go_home()
    circuit.convert_to_ndarray()

    return circuit
