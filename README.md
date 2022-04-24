| [ArXiv](https://arxiv.org/abs/2002.02515)|

# Network Equivalency
This part includes implementations of eight equivalent networks in light of the extended De Morgan's law. Eight networks are shown in Figure 1. The purpose of this experiment is to evaluate if the performance of these eight networks is close to one another.

<p align="center">
  <img width="480" src="https://github.com/FengleiFan/Duality/blob/master/equivalent_networks.png">
</p>

<p align="center">
  Figure 1. Eight equivalent networks in light of the extended De Morgan's law.
</p>


## Folders ## 
**NetworkEquivalence**: this directory contains codes for eight equivalent networks. The used dataset is the breast cancer datatset.<br/>


##  Running Experiments ## 

Please first go to each directory. Each directory consists of two scripts. One is about the network, and the other is the main file.  

```ruby
>> python NetworkEquivalence/NetworkEquivalency_I.py    
>> python NetworkEquivalence/NetworkEquivalency_II.py 
>> python NetworkEquivalence/NetworkEquivalency_III.py 
```

# Robustness
In this part, we compare the robustness of a deep and a wide quadratic network constructed as Figure 2 shows.
<p align="center">
  <img width="480" src="https://github.com/FengleiFan/Duality/blob/master/quadratic_networks.png">
</p>

<p align="center">
  Figure 2. The width and depth equivalence for networks of quadratic neurons. In this construction, a deep network is to implement the continued fraction of a polynomial, and a wide network reflects the factorization of the polynomial.
</p>

