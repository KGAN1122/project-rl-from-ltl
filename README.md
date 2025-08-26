# Sample-Efficient Reinforcement Learning for Robotics with High-Level Task Specifications

## Abstract
This work studied reinforcement learning under Linear Temporal Logic (LTL) constraints, con- sidering optimizations from both model-free and model-based perspectives. On the model-free side, we applied the K-counter reward structure with counterfactual imagining (CF+KC) and further examined the Stochastic Ensemble Value Expansion (STEVE) algorithm. The experiments show that both methods can improve sample efficiency to some extent. We then explored the integration of model-free and model-based methods. The results indicate that combining CF+KC with R-max (optimistic initialization model-based reinforcement learning) achieves the best balance, delivering superior performance in terms of both runtime efficiency and stability.

## Dependencies
The followings are the essential tools and libraries to run our RL algorithms:
- [Python](https://www.python.org/): (>=3.7)
- [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` 
Download from https://www7.in.tum.de/~kretinsk/rabinizer4.html and follow instructions to add ```ltl2ldba``` to ```PATH```

To run experiments, use main.py. The --task option is required and can be used to choose the task from the following: 'frozen_lake' and 'office_world'.

```shell
$ python main.py --task 'frozen_lake'
$ python main.py --task 'office_world'
```
