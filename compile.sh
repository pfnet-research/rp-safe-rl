g++ -std=c++1z -Ofast -shared -fPIC `python3 -m pybind11 --includes` circuit/model_control.cpp -o circuit/mpc`python3-config --extension-suffix` -fopenmp
g++ -std=c++1z -Ofast -shared -fPIC `python3 -m pybind11 --includes` jam/model_control.cpp -o jam/mpc`python3-config --extension-suffix` -fopenmp
g++ -std=c++1z -Ofast circuit/sampler.cpp -o circuit/sampler
g++ -std=c++1z -Ofast jam/wall_estimator.cpp -o jam/wall
g++ -std=c++1z -Ofast jam/car_estimator.cpp -o jam/car
