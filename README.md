# Large-Scale_FL
This is an implementation of the seminal algorithm federated learning algorithm FedAvg described in the AISTATS 2017 paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) by McMahan et al.

# Implementation details
The main steps are as follows:
1) A centralized server coordinates several clients. The clients are the devices which contain the data and actually carry out the training and backpropagation.
2) At the beginning of each round, the server sends it's version of the model weights to all devices. Each device trains these weights on it's own subset of the dataset (in this case IID shards of MNIST).
3) After a certain number of updations, all devices send their local version of the weights (now updated by gradient) back to the server.
4) The server essentially scales these received weights by a factor proportional to the number of clients and aggregates them.
5) The weights of this new aggregated model are then sent out to all the clients again in the next round.
6) This process repeats till the final aggregated model converges in expectation.

```FedAvg.ipynb``` simulates this process by using several independent instances of the model to represent the clients and the server. 

# Acknowledgement

Major credits for the implementation details go to https://github.com/eceisik/ece_fl_public <br>
Also check out the original paper at [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
