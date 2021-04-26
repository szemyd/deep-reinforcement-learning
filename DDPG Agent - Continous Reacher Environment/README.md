[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Deep Deterministic Policy Grient Agent to solve the Reacher environment

This repo contains implementation of a DDPG Agent in __PyTorch__.

Agents are tested on the Single [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment packaged and provided by Udacity.

### Table of contents
* [Environment](#environment)
    * [Goal & Reward](#goal-reward)
    * [Properties of the Environment](#properties-of-the-environment)
    * [Installing Python Environment](installing-python-environment)
    * [Quick Setup of Environment](#quick-setup-of-environment)
    * [Quick Initialization of Environment](#quick-initialization-of-environment)
* [Agent](#agent)
    * [Hyperparameters](#hyperparameters)
    * [How to train your agent](#how-to-train-your-agent)
    * [How to test your agent](#how-to-test-your-agent)

## Environment
---
![Trained Agent][image1]
</br>
</br>

### Goal & Reward
Keep the end of the effector in the given moving goal sphere. Interaction is episodic (but could be continuous), with a hard step limit of 1000.
</br>
</br>

### Properties of the environment
|                |        | 
| -------------- | ------ |
| _state space_: | __33__ (position, rotation, velocity and angular velocity) |
| _action space_: | __4__ (continuous, all between -1 and 1) |
| _agents (brains)_: | __1__ |
| _considered solved_: | __> +30.0__ avg. over 100 episodes |
| _termination criteria_:| __1000__ time steps | 
| _reward_:| __+0.1__ at each time step when arm is in the goal sphere|



_source of the environment:_ __Udacity - Deep Reinforcement Learning__
_engine_: __unityagents__ `from unityagents import UnityEnvironment`
</br>
</br>
</br>

### Installing Python environment

Python environment can be set up as follows, as per the [Udacity DRLND guide](https://github.com/udacity/deep-reinforcement-learning#dependencies):

1. Create (and activate) a new environment with Python 3.6.

Linux or Mac:
```python
    conda create --name drlnd python=3.6
    source activate drlnd
```
Windows:
```python
    conda create --name drlnd python=3.6 
    activate drlnd 
```

2. Follow the instructions in this repository to perform a minimal install of OpenAI gym.

- Next, install the classic control environment group by following the instructions here.
- Then, install the box2d environment group by following the instructions here.

3. Clone the repository (if you haven't already!), and navigate to the python/ folder. Then, install several dependencies.

```python
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
```

4. Create an IPython kernel for the drlnd environment.
```python
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

More info on the original Udacity [DRLND repo](https://github.com/udacity/deep-reinforcement-learning).
</br>
</br>

### Quick setup of Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

</br>
</br>



## Agent
---
### How to train your agent
After you have downloaded the environment executable and installed python with the relevant packages, navigate to t_continous_control.ipynb jupyter file  (jupyter comes with anaconda, but in case you don't have it, you can find it [here](https://jupyter.org/#:~:text=The%20Jupyter%20Notebook%20is%20an,machine%20learning%2C%20and%20much%20more.).

In __t_continous_control.ipynb__ execute all cells above 'Display and Saved'. 
Then you have two options:
* Display and save a figure and scores to _Experiments_ directory.
  * Load previous scores and network and continue training
  * Watch a trained agent perform




</br>
</br>

### How to test your agent
1. If you haven't trained the agent since the new kernel was initialized, you have to load a previously trained agent
2. Check if there is a saved agent. In the name of the saved agent you will see the hidden layer parameters of that agent. 
3. In constants.py set the parameter of the agent to match with the saved file.
4. Restart kernel and execute the first two cells in __t_continous_control.ipynb__ to load in the changed constants
5. Run all cells under _See how your agent performs_

If the network specifications in _constants_ don't match that of the saved agent, you won't be able to load it.
</br>
</br>

### Hyperparameters

__Actor Network Architecture__
|     Input Layer size         |     33                 |
|------------------------------|------------------------|
|     Hidden Layer size 1      |     160                |
|     Hidden   Layer size 2    |     160                |
|     Output Layer size        |     4 (action-size)    |
|     Activation   function    |     ReLU               |
|     Output function          |     tanh               |

</br>

__Critic Network Architecture__

|     Input Layer size                              |     33      |
|---------------------------------------------------|-------------|
|     Input Layer size                              |     33      |
|     Hidden Layer size 1                           |     160     |
|     Action   size (appended to hidden layer 1)    |     4       |
|     Hidden Layer size 2                           |     160     |
|     Output   Layer size                           |     1       |
|     Activation function                           |     ReLU    |
|     Output   function                             |     -       |
</br>

__Training Hyperparameters__
|     Input Layer size                                     |     33      |
|----------------------------------------------------------|-------------|
|     TAU     (soft update of target network parameter)    |     1e-4    |
|     Learning   Rate                                      |     1e-4    |
|     Target network is updated every {} episode:          |     1       |
|     MU   (Noise)                                         |     0.0     |
|     Theta (Noise)                                        |     0.15    |
|     Sigma   (Noise)                                      |     0.05    |
|     Random seed                                          |     1       |



