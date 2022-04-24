# Network Equivalency
| [ArXiv](https://arxiv.org/abs/2002.02515)|

This part includes implementations of eight equivalent networks in light of the extended De Morgan's law.

<p align="center">
  <img width="480" src="https://github.com/FengleiFan/Duality/blob/master/equivalent_networks.png">
</p>

<p align="center">
  Figure 1. Eight equivalent networks in light of the extended De Morgan's law.
</p>

<p align="center">
  <img width="320" src="https://github.com/FengleiFan/ReLinear/blob/main/Figure_guaranteed_improvements.png">
</p>

<p align="left">
  Figure 2. The performance of a quadratic network trained using the proposed ReLinear stragetgy, with an observed improvement than the conventional network of the same structure. $(\gamma_g,\gamma_b)$, $(\alpha_g,\alpha_b)$, and $(\beta_g,\beta_b)$ are hyperparameters of ReLinear. As these hyperparameters increases from 0, the trained model transits from the conventional model to the quadratic, and the model's performance reaches the optimality.
</p>

## Folders 
**TrainCompactQuadraticNetworksViaReLinear**: this directory contains the implementation of ReLinear on a compact quadratic network. The compact quadratic network consists of compact quadratic neurons that simplify the quadratic neuron by eradicating interaction terms. The used dataset is CIFAR10.<br/>
**TrainQuadraticNetworksViaReLinear+ReZero**: this directory contains implementations of ReLinear+ReZero on a quadratic network. Because the quadratic network is based on the redidual connection, we can combine the proposed ReLinear with [ReZero](https://arxiv.org/pdf/2003.04887.pdf) that was devised for training residual networks. The used dataset is CIFAR10. <br/>
**TrainQuadraticNetworksViaReLinear**: this directory contains the implementation of ReLinear on a quadratic network. The used dataset is CIFAR10.<br/>


## Running Experiments

Please first go to each directory. Each directory consists of two scripts. One is about the network, and the other is the main file.  

```ruby
>> python TrainCompactQuadraticNetworksViaReLinear/qresnet_smaller.py           
>> python TrainQuadraticNetworksViaReLinear+ReZero/Rezero_train_56.py    
>> python TrainQuadraticNetworksViaReLinear/qtrainer_10_5.py        
```

