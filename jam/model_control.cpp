#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Complex = complex<double>;

random_device rnd;
mt19937_64 mt(rnd());

struct Action {
    double acc, omega, prob;
};

vector<Action> my_action_list;
vector<Action> other_action_list;
constexpr double penalty = 50;

struct Agent {
    Agent(Complex pos, double radius, Complex front, double v)
        : m_pos(pos), m_radius(radius), m_front(front), m_v(v) {}

    Complex m_pos;
    double m_radius;
    Complex m_front;
    double m_v;

    double roll(double x, double mi, double ma) {
        if (x > ma) return ma;
        if (x < mi) return mi;
        return x;
    }

    Agent next_agent(Action action) {
        Complex next_pos = m_pos + m_front * m_v;
        double next_v = this->roll(m_v + action.acc, -0.1, 0.1);
        Complex next_front = m_front * exp(Complex(0, action.omega));
        return Agent(next_pos, m_radius, next_front, next_v);
    }
};

using RA = pair<double, Agent>;

struct OtherCar {
    OtherCar(Complex pos, Complex front, double v)
        : m_pos(pos), m_front(front), m_v(v) {}
    Complex m_pos;
    Complex m_front;
    double m_v;

    double roll(double x, double mi, double ma) {
        if (x > ma) return ma;
        if (x < mi) return mi;
        return x;
    }

    OtherCar next_agent(Action action) {
        Complex next_pos = m_pos + m_front * m_v;
        double next_v = this->roll(m_v + action.acc, 0, 0.06);
        Complex next_front = m_front * exp(Complex(0, action.omega));
        if (next_pos.imag() > 10000 or next_pos.real() > 10000) exit(1);
        return OtherCar(next_pos, next_front, next_v);
    }
};

struct Predictor {
    Predictor(vector<Complex> wall, int depth)
        : m_wall(wall), m_max_depth(depth) {}

    vector<Complex> m_wall;
    int m_max_depth;

    double progress(Agent agent) {
        return -int(abs(Complex(2.6, 3.0) - agent.m_pos) * 20);
    }

    bool crashed(Agent agent) {
        vector<double> arr(m_wall.size());
        for (int i = 0; i < int(arr.size()); i++) {
            arr[i] = abs(m_wall[i] - agent.m_pos);
        }
        double min = *min_element(arr.begin(), arr.end());
        return min < agent.m_radius;
    }

    // not about other cars
    RA next_r_and_agent(Agent agent, Action action) {
        double r = -0.05;
        double prev = progress(agent);
        Agent next_agent = agent.next_agent(action);
        double now = progress(next_agent);

        r += (now - prev);
        if (crashed(next_agent)) r -= penalty;
        if (abs(agent.m_v) < 0.005) r -= 0.05;
        if (next_agent.m_pos.imag() > 3) r += 10;

        return RA(r, next_agent);
    }

    vector<double> rewards(Agent agent, OtherCar car) {
        vector<double> vec(my_action_list.size());

#pragma omp parallel for
        for (int i = 0; i < int(my_action_list.size()); i++)
            vec[i] = max_V_sa(agent, my_action_list[i], car, 0);

        return vec;
    }

    double max_V_s(Agent agent, OtherCar car, int depth) {
        if (abs(agent.m_pos - car.m_pos) < agent.m_radius * 2) return -penalty;
        if (depth == m_max_depth) return 0;

        vector<double> val(my_action_list.size());
        for (int i = 0; i < int(val.size()); i++)
            val[i] = max_V_sa(agent, my_action_list[i], car, depth);

        return *max_element(val.begin(), val.end());
    }

    double max_V_sa(Agent agent, Action action, OtherCar car, int depth) {
        RA ra = next_r_and_agent(agent, action);
        double r = ra.first / 8;
        Agent next_agent = ra.second;
        if (r < -20) return r;
        if (next_agent.m_pos.imag() > 3) return r;

        double val = 0;
        for (Action a : other_action_list) {
            double tmp = max_V_s(next_agent, car.next_agent(a), depth + 1);
            val += a.prob * tmp;
        }

        return r + val;
    }
};

using P = pair<double, double>;

int func(int depth, double agent_v, Complex pos, Complex front,
         vector<double> other_v, vector<Complex> other_pos,
         vector<Complex> other_front, vector<Complex> wall) {
    // make action lists
    my_action_list = vector<Action>();
    other_action_list = vector<Action>();

    vector<P> acc{{-0.02, 0}, {0, 0}, {0.02, 0}};
    vector<P> omega{{-0.3, 0}, {-0.1, 0}, {0, 0}, {0.1, 0}, {0.3, 0}};
    for (P a : acc)
        for (P w : omega)
            my_action_list.emplace_back(
                Action{a.first, w.first, a.second * w.second});

    vector<P> acc2{{-0.02, 0.2}, {0, 0.6}, {0.02, 0.2}};
    vector<P> omega2{
        {-0.15, 0.2}, {-0.05, 0.2}, {0, 0.2}, {0.05, 0.2}, {0.15, 0.2}};

    for (P a : acc2)
        for (P w : omega2)
            other_action_list.emplace_back(
                Action{a.first, w.first, a.second * w.second});

    vector<OtherCar> cars;
    for (int i = 0; i < int(other_v.size()); i++)
        cars.emplace_back(OtherCar{other_pos[i], other_front[i], other_v[i]});

    double agent_r = 0.1 + 1e-4;

    Agent agent(pos, agent_r, front, agent_v);

    Predictor predictor(wall, depth);

    vector<double> ans(my_action_list.size());
    for (OtherCar car : cars) {
        vector<double> re = predictor.rewards(agent, car);
        for (int i = 0; i < int(re.size()); i++) ans[i] += re[i];
    }

    auto ma = *max_element(ans.begin(), ans.end());

    vector<double> vec;
    for (int i = 0; i < ans.size(); i++)
        if (ans[i] == ma) vec.push_back(i);

    uniform_int_distribution<> indice(0, vec.size() - 1);

    return vec.at(indice(mt));
}

PYBIND11_MODULE(mpc, m) {
    m.doc() = "mpc";
    m.def("calc", &func, "calculate threat value for each state action pair");
}
