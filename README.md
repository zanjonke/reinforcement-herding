# Reinforcement herding

## Project description

We were interested in tackling the shepherding problem first introduced by Strömbom et al. [[1]](#literature). It describes a field filled with sheep, that behave as a swarm system, and a sheepdog, whose task is to herd the sheep to a certain point, an enclosure. To solve the problem we must find an appropriate behaviour pattern for the sheepdog, for it to successfully shepherd the sheep into an enclosure. We think this problem could be tackled by using reinforcement learning, where the sheepdog is an agent, unfamiliar with the herding characteristics of the sheep. Our end result would be a trained agent capable of solving this task.

The goal of this project was to develop a reinforcement learning model which is able to solve the shepherding problem. We decided to tackle the problem with DQN, which is a deep neural network, designed estimate values of actions in given states. To take into account the inherent property of the problem, that is the subtasks of collecting and driving the sheep, we trained two models, one for each.

## Repository description

The main files and folders are the following: 

```bash
.
├── double_testing.py
├── double_training.py 
├── models 
├── sheepherding.py 
├── testing.py 
└── training.py  
```

Where
- **_double_testing.py_** implements the testing framework for hierarchical model ie. one model for collecting and another for driving
- **_double_training.py_** implements the training framework for hierarchical model ie. one model for collecting and another for driving
- **_models_** contains DQN models stored during training
- **_sheepherding.py_** implements the Strombom et. al. sheepherding environment
- **_testing.py_** implements the basic DQN framework for solving environments
- **_training.py_** implements the testing framework for basic DQN framework

# Literature

[1] Strömbom D., Mann R. P., Wilson A. M., Hailes S., Morton A. J., Sumpter D. J. T. and King A. J.,
2014, Solving the shepherding problem: heuristics for herding autonomous, interacting agentsJ. R. Soc. Interface.112014071920140719
http://doi.org/10.1098/rsif.2014.0719
