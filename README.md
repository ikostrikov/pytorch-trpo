# PyTorch implementation of TRPO

Try my implementation of [PPO](github.com/ikostrikov/pytorch-a2c-ppo-acktr/) (aka newer better variant of TRPO), unless you need to you TRPO for some specific reasons.

##

This is a PyTorch implementation of ["Trust Region Policy Optimization (TRPO)"](https://arxiv.org/abs/1502.05477).

This is code mostly ported from [original implementation by John Schulman](https://github.com/joschu/modular_rl). In contrast to [another implementation of TRPO in PyTorch](https://github.com/mjacar/pytorch-trpo), this implementation uses exact Hessian-vector product instead of finite differences approximation.

## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage

```
python main.py --env-name "Reacher-v1"
```

## Recommended hyper parameters

InvertedPendulum-v1: 5000

Reacher-v1, InvertedDoublePendulum-v1: 15000

HalfCheetah-v1, Hopper-v1, Swimmer-v1, Walker2d-v1: 25000

Ant-v1, Humanoid-v1: 50000

## Results

More or less similar to the original code. Coming soon.

## Todo

- [ ] Plots.
- [ ] Collect data in multiple threads.
