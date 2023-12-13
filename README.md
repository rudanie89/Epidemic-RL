# EPIDEMIC - RL
Epidemic-RL is an RL algorithm to select a subset of debunkers to spread the true information for combating rumour spread on OSNs.
This repository is for the paper "Mitigation of Rumours in Social Networks via Epidemic Model-based Reinforcement Learning" in DSAA2022.

# Paper Abstract
While detection of rumours in online social networks has been intensively studied in the literature, mitigation of the spread of rumours has only recently gained attention and remains a challenging task. Some studies developed user opinion models to find top influential users as debunkers to spread the truth to counter rumour spread. Other studies designed an intervention framework to optimize the mitigation activities for given debunkers. The issue of optimizing the selection of debunkers in a dynamic environment where users’ beliefs and behaviour change remains under investigated. This paper addresses this issue by proposing a rumour mitigation approach based on the deep reinforcement learning framework. In particular, we model the changes in users’ beliefs with an epidemic model. We further employ deep reinforcement learning to train an agent to learn a multi-stage policy for selecting the optimal debunkers to inject truthful information under a budget constraint. Our model selects debunkers to inject truthful information at multiple stages with an overall objective to maximize the number of users who will believe in the true information (a.k.a number of recovered nodes), such that the spread of rumours is minimized. Our experiments on synthetic and real-world social networks show that our proposed method for rumour mitigation can effectively minimize the spread of rumours.

# Dataset
## Synthetic dataset
## Real dataset

# Requirements

# To train the model
Run the following command:
```bash
$ python DQN.py

# References

# Citation
if you find this code and paper are useful, please cite our paper.
@inproceedings{nie2022mitigation,
  title={Mitigation of Rumours in Social Networks via Epidemic Model-based Reinforcement Learning},
  author={Nie, H Ruda and Zhang, Xiuzhen and Li, Minyi and Dolgun, Anil},
  booktitle={2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA)},
  pages={1--10},
  year={2022},
  organization={IEEE}
}
