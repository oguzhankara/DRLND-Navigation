[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://cdn-images-1.medium.com/max/1600/1*2wOzh6K4NMMrWYvZ0G5KUA.png "Deep Q-Learning Algorithm"


# Navigation
### Introduction
The goal for this project is to train an agent to navigate and collect yellow bananas as well as avoiding blue bananas in a large, square world. The environment is based on **Unity ML-agents**.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Learning Algorithm and Agent Implementation

**Deep Q-Learning** Algorithm lies in the heart of the approach for the agent implementation. This algorithm is a value-based policy and it is composed of a RL method called SARSA max and uses local and target deep neural networks for action-values approximation.

In the agent implementation; **Experience Replay** and **Fixed Q-Targets** improvements are used. The details for these improvements can be analyzed in Deepmind's [Nature publication : "Human-level control through deep reinforcement learning (2015)"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

## *The pseudocode of the Deep Q-Learning Algorithm can be seen below:*
![Deep Q-Learning Algorithm][image2]



### Optimized DQN Hyper Parameters and Neural Network

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.97            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
HIDDEN_LAYERS = [64,64] # Network fc layers
eps_start=1.0           # Start value of epsilon value in the beginning of the training
eps_min=0.01            # Minimum value for the epsilon to be reached
eps_decay=0.995         # Epsilon decay value updated after each episode e.g. eps = max(eps * eps_decay, eps_min)
```


**Neural Network Architecture:**
```
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (actions): Linear(in_features=64, out_features=4, bias=True)
)

Optimizer: Adam Optimizer

Activation Function: Rectified Linear Units (ReLU) activation function
```



### Training Results

The environment is solved in 525 episodes with an average score of 13. Total training time with Nvidia K80 GPU is 14.6 minutes.

```
Episode 100	Average Score: 1.15
Episode 200	Average Score: 4.51
Episode 300	Average Score: 8.88
Episode 400	Average Score: 10.81
Episode 500	Average Score: 11.63
Episode 600	Average Score: 12.48
Episode 625	Average Score: 13.00
Environment solved in 525 episodes!	Average Score: 13.00
Total Training time = 14.6 min
```

![Training Score](images/Score.png)


### Ideas For Future Network

Following improvements can be implemented to further increase the general performance;
1. Prioritized Experience Replay
2. Double DQN
3. Dueling DQN
4. Learning from raw pixels by conducting CNN instead of environment's internal vector states (37 dimensions). This approach is expected to bring some preprocessing cost.

Apart from the possible improvements; the variability of scores is an issue and needs to be analyzed and improved.
