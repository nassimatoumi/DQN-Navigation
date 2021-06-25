# Report

## Implementation
In this implementation, the DQN main and target networks are implemented using PyTorch. The neural networks comprised two hidden layers of 128 and 64 units respectively, with ReLu activation functions, and the activation function of the last layer being softmax.
The solution implements experience replay, to break correlation between the experiences, and thus avoid bias during learning, the experiences are stored in a buffer of a size of 100000, and randomly sampled in batches of 64.
The neural network weights are updated using the ADAM optimizer, with a learning rate of 1e-3, and a discount factor gamma of 0.99. Finally, the target network is updated every 4 episodes with a soft update using a tau value of 1e-3.

## Results
The model was trained on a GPU, and was able to obtain an average reward of over a 13.0 for 100 consecutive episodes after 1279 episodes, as shown in the figure below.

![alt text](https://github.com/nassimatoumi/DQN-Navigation/blob/ba1b6265e371a34dcb97fe383aa08e014fe706e2/Scores.png)

## Future works
For future works, additional methods should be implemented such as Double DQN, and Prioritized Experience Replay, which should allow the model to converge more quickly.
