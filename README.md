# gPC_toolbox
```
A generalized Polynomial Chaos Toolbox for projecting Chance-Constrained Stochastic Optimal Control Problem
to a Deterministic Optimal Control Problem
1) sde to ode dynamics 
2) expectation cost to deterministic cost
3) linear chance constraints to second-order cone  
4) quadratic to semi-definite-program
5) gpc --> confidence level 
6) initial and terminal constraints
```
## Dependencies
```
1) Numpy 
2) Scipy
3) Sympy
4) Matplotlib
5) Cloudpickle
```
## How to run 
```
cd gPC_toolbox
# Generate data using 
python3 test_gpc_pendulum.py
# Test propagation  
python3 test_mc_kf_gpc_comparision.py
```

## Reference 
```
@INPROCEEDINGS{nakka2019trajectory,
  author={Y. K. {Nakka} and S. {Chung}},
  booktitle={2019 IEEE 58th Conference on Decision and Control (CDC)}, 
  title={Trajectory Optimization for Chance-Constrained Nonlinear Stochastic Systems}, 
  year={2019},
  volume={},
  number={},
  pages={3811-3818},
  doi={10.1109/CDC40024.2019.9028893}}
```
