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
    double acc, omega;
};

struct Agent {
    Agent(Complex pos, double radius, Complex front, double v,
          vector<Action> action_list)
        : m_pos(pos),
          m_radius(radius),
          m_front(front),
          m_v(v),
          m_action_list(action_list) {}

    Complex m_pos;
    double m_radius;
    Complex m_front;
    double m_v;
    vector<Action> m_action_list;

    double roll(double x, double mi, double ma) {
        if (x > ma) return ma;
        if (x < mi) return mi;
        return x;
    }

    Agent next_agent(Action action) {
        Complex next_pos = m_pos + m_front * m_v;
        double next_v = this->roll(m_v + action.acc, -0.04, 0.1);
        Complex next_front = m_front * exp(Complex(0, action.omega));
        return Agent(next_pos, m_radius, next_front, next_v, m_action_list);
    }
};

using RA = pair<double, Agent>;

struct Predictor {
    Predictor(Agent agent, vector<Complex> wall, vector<Complex> cpos,
              int depth, double safety_ratio)
        : m_first_agent(agent),
          m_wall(wall),
          m_cpos(cpos),
          m_max_depth(depth),
          m_safety_ratio(safety_ratio) {}

    Agent m_first_agent;
    vector<Complex> m_wall;
    vector<Complex> m_cpos;
    int m_max_depth;
    double m_safety_ratio;

    int progress(Agent agent) {
        vector<double> arr(m_cpos.size());
        for (int i = 0; i < int(arr.size()); i++) {
            arr[i] = abs(m_cpos[i] - agent.m_pos);
        }
        auto min = min_element(arr.begin(), arr.end());
        return int(distance(arr.begin(), min));
    }

    bool crashed(Agent agent) {
        vector<double> arr(m_wall.size());
        for (int i = 0; i < int(arr.size()); i++) {
            arr[i] = abs(m_wall[i] - agent.m_pos);
        }
        double min = *min_element(arr.begin(), arr.end());
        return min < agent.m_radius * this->m_safety_ratio;
    }

    RA next_r_and_agent(Agent agent, Action action) {
        double r = 0;
        int prev = progress(agent);
        Agent next_agent = agent.next_agent(action);
        int now = progress(next_agent);
        if (abs(now - prev) < m_cpos.size() / 4) r += now - prev;
        if (crashed(next_agent)) r -= 200;
        if (abs(agent.m_v) < 0.005) r -= 1;

        return RA(r, next_agent);
    }

    vector<double> action_values() {
        vector<double> vec(m_first_agent.m_action_list.size());

#pragma omp parallel for
        for (int i = 0; i < int(m_first_agent.m_action_list.size()); i++)
            vec[i] = max_V_sa(m_first_agent, m_first_agent.m_action_list[i], 1);

        return vec;
    }

    double max_V_s(Agent agent, int depth) {
        double max = -50000;
        for (Action action : agent.m_action_list) {
            double v = max_V_sa(agent, action, depth);
            if (v > max) max = v;
        }
        return max;
    }

    double max_V_sa(Agent agent, Action action, int depth) {
        RA ra = next_r_and_agent(agent, action);
        double r = ra.first;
        Agent next_agent = ra.second;
        if (r < -100 or depth == m_max_depth) return r;
        double v = max_V_s(next_agent, depth + 1);
        return r + v;
    }
};

int func(int depth, double agent_v, Complex pos, Complex front,
         vector<Complex> wall, vector<Complex> cpos) {
    vector<double> acc{-0.02, 0, 0.02};
    vector<double> omega{-0.15, -0.05, 0, 0.05, 0.15};
    vector<Action> list;
    for (double a : acc)
        for (double w : omega) list.emplace_back(Action{a, w});

    double agent_r = 0.1;
    double safety_ratio = 1;

    Agent agent(pos, agent_r, front, agent_v, list);

    Predictor predictor(agent, wall, cpos, depth, safety_ratio);

    vector<double> action_values = predictor.action_values();

    auto ma = *max_element(action_values.begin(), action_values.end());

    vector<int> vec;
    for (int i = 0; i < action_values.size(); i++)
        if (action_values[i] == ma) vec.push_back(i);

    uniform_int_distribution<> indice(0, vec.size() - 1);

    cout << "max = " << ma << endl;
    return vec.at(indice(mt));
}

PYBIND11_MODULE(mpc, m) {
    m.doc() = "mpc";
    m.def("calc", &func, "calculate threat value for each state action pair");
}