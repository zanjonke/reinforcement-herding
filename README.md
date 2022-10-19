# reinforcement-herding
Learning the herding task using reinforcement learning

## Project description

We are interested in tackling the shepherding problem first introduced by Strömbom et al. [1]. It describes a field filled with sheep, that behave as a swarm system, and a sheepdog, whose task is to herd the sheep to a certain point, an enclosure. To solve the problem we must find an appropriate behaviour pattern for the sheepdog, for it to successfully shepherd the sheep into an enclosure. We think this problem could be tackled by using reinforcement learning, where the sheepdog is an agent, unfamiliar with the herding characteristics of the sheep. Our end result would be a trained agent capable of solving this task.

As a starting point we would take the FRIsheeping source code and refactor it to suit our needs for training our agent, and get familiar with what data can we feed our agent. We will also need to search related works for similiar approaches, to see what worked in other implementations, and decide on a learning algorithm appropriate for our environment.

## Rough plan ahead

### 1st deadline
- take a look into related works
- get familiar with the FRIsheeping source code
- concretely define the goals of the project

### 2nd deadline
- adjust goals according to feedback received from first report
- 1st version of the reinforcement learning algorihtm

### 3rd deadline
- adjust goals according to feedback received from second report
- polishing the reinforcement learning algorithm
- video creation


# Literature
[1] Strömbom D., Mann R. P., Wilson A. M., Hailes S., Morton A. J., Sumpter D. J. T. and King A. J.,
2014, Solving the shepherding problem: heuristics for herding autonomous, interacting agentsJ. R. Soc. Interface.112014071920140719
http://doi.org/10.1098/rsif.2014.0719
