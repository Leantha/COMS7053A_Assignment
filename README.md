# COMS7053A_Assignment
# MiniHack the  Planet
This repository contains 2 reinforcement learning implementations for exploring the Minihack-Quest-Hard-v0 enviroment. The 2 agents used for this assignment are derived from reinformcement learning methods Deep-Q Network or DQN and REINFORCE. From these 2 approaches, the REINFORCE approach achieves better results and is able to actually do some exploring in the enviroment whilst the DQN agent performs dismally. REINFORCE is therefore over DQN for this project.

More details about the methods and agents can be found in the project report along with results from the runs.

# Installation
Follow the directions given on the GitHub repository for obtaining the MiniHack package.\ https://github.com/facebookresearch/minihack \
Because MiniHack is essentially a wrapper for the NetHack Learning Environment (NLE), it may be useful to first install the NLE package from https://github.com/facebookresearch/nle.

Once the project is cloned, please note that you will need to change the file paths in the DQN/Agent.py and Reinforce/reinforce.py files to a specified path on your PC inorder to collect the output files of each experiment.

# To run the DQN agent
The DQN agent can be run by going to the location of the file and running the following command in terminal:
```
python3 Agent.py
```

# To run the REINFORCE agent
To train the reinforce agent, you can simply go the the location of the reinforce code and run the command in terminal below:
```
python3 reinforce.py
```
There is a chance that you may get numpy() errors on the array function. This can be resolved by following the path in your anaconda3 library folder; ~/anaconda3/lib/python3.8/site-packages-torch/_tensor.py and making the following changes:
```python
    def __array__(self, dtype=None):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        if dtype is None:
            return self.detach().numpy()
        else:
            return self.detach().numpy().astype(dtype, copy=False)
```

# Developed by:
Leantha Naicker (788753) \
Clerence Mathonsi (2512711)
