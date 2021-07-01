# DQN-Navigation


## Description

In this project, a Deep Reinforcement Learning agent is trained to navigate a square world while collecting a maximal number of yellow bananas, and avoiding blue bananas. The environment returns a reward of +1 for each collected yellow banana, and a penalty of -1 for blue bananas. The state space contains has 37 dimensions, containing the agent's velocity, and the observed objects around the agent depending on its direction.
Using that information, the agent can perform 4 different actions:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and the agent is trained once the average score has reached +13 for over 100 consecutive episodes.

### Getting Started: 
To run this project, multiple dependencies should be met. In particular, the environment should be installed. 

Depending on the host operating system, the environment can be downloaded from one of the following links:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
Afterwards, the file should be decompressed, and the path for the environment file should be edited on the **train_model.py** and **use_model.py** files, at line 9. 

### Instructions:
- To train the agent from scratch, run the following command, while specifying in the file the path of the newly created model if needed:

    **python3 train_model.py**

Note that the hyperparameter values can be tuned in the dqn_agent file, and the Deep Neural Network architecture can be changed in the file model.py 

- To use the trained model, run the following command:

    **python3 use_model.py**
    
The path of the model file can be specified in line 8. 
