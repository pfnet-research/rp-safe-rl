#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Complex = complex<double>;
using RV = pair<Complex, double>;
constexpr double one_deg = M_PI / 180;

random_device rnd;
mt19937_64 mt(rnd());
uniform_real_distribution<> rx_(-0.7, 0.7);
uniform_real_distribution<> ry_(-0.3, 0.3);
uniform_real_distribution<> v_(-0.1, 0.1);
uniform_real_distribution<> dice_(0, 1);
uniform_int_distribution<> action_num(0, 14);
constexpr double agent_r = 0.1;

struct Action {
    double acc, omega;
};

double action_dice() {
    double dice = dice_(mt), level = 0;

    level += 0.2;
    if (dice < level) return -0.02;
    level += 0.6;
    if (dice < level) return 0;
    return 0.02;
}

double omega_dice() {
    double dice = dice_(mt), level = 0;

    level += 0.2;
    if (dice < level) return -0.3;
    level += 0;
    if (dice < level) return -0.1;
    level += 0.6;
    if (dice < level) return 0;
    level += 0;
    if (dice < level) return 0.1;
    return 0.3;
}

struct State {
    State(Complex r, double v) : m_r(r), m_v(v) {}

    double roll(double x, double mi, double ma) {
        if (x > ma) return ma;
        if (x < mi) return mi;
        return x;
    }

    State next_state(double acc, double omega) {
        Complex rotate = exp(Complex(0, -omega));
        Complex next_r = (m_r - m_v) * rotate;

        double next_v = this->roll(m_v + acc, -0.1, 0.1);
        return State(next_r, next_v);
    }

    double min_distance() { return abs(m_r); }

    bool is_danger(double agent_r) { return abs(m_r) < agent_r; }

    Complex m_r;
    double m_v;
};

double threat(State now_state, double acc, double omega, const int depth,
              const double beta) {
    if (now_state.is_danger(agent_r)) return 1;

    for (int t = 0; t < depth; t++) {
        if (t != 0) acc = action_dice(), omega = omega_dice();
        now_state = now_state.next_state(acc, omega);
        if (now_state.is_danger(agent_r)) return pow(beta, t + 1);
    }
    return 0;
}

double observed_rad(Complex obj) {
    double rad = arg(obj);
    for (double th = -M_PI;; th += one_deg)
        if (th < rad and rad <= th + one_deg) {
            return th + one_deg / 2;
        }
}

vector<Action> list;

void sample(const int depth, const int samples, double beta, ofstream& ofs) {
    double rx = rx_(mt), ry = ry_(mt), v = v_(mt);
    Complex obj = Complex(rx, ry);
    State ini_state(obj, v);

    Action action = list[action_num(mt)];
    double threat_value = 0;

    vector<double> threat_array(samples);

#pragma omp parallel for
    for (int i = 0; i < samples; i++)
        threat_array[i] =
            threat(ini_state, action.acc, action.omega, depth, beta);

    threat_value =
        accumulate(begin(threat_array), end(threat_array), 0.0) / samples;

    double rad = observed_rad(obj);
    double norm = abs(obj);

    ofs << norm * cos(rad) << " " << norm * sin(rad) << " " << v << " "
        << " " << action.acc << " " << action.omega << " " << threat_value
        << '\n';

    ofs << norm * cos(rad) << " " << -norm * sin(rad) << " " << v << " "
        << " " << action.acc << " " << -action.omega << " " << threat_value
        << '\n';
}

int main(int, char** argv) {
    const int depth = atoi(argv[1]);
    int times = atoi(argv[2]);
    int samples = atoi(argv[3]);
    double beta = atof(argv[4]);

    vector<double> acc{-0.02, 0, 0.02};
    vector<double> omega{-0.3, -0.1, 0, 0.1, 0.3};
    for (double a : acc)
        for (double w : omega) list.emplace_back(Action{a, w});

    ofstream ofs("output.dat");

    const int chunk = 1000;
    for (int i = 0; i < times; i++) {
        if (i % chunk == 0 and i)
            cout << i / chunk << "k data finished" << endl;
        sample(depth, samples, beta, ofs);
    }

    ofs.close();

    return 0;
}
