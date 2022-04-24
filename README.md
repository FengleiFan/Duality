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

Please first go to the directory of "NetworkEquivalence" .

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

**Experimental Design** 

We first preprocessed the MNIST dataset using image deskewing and dimension deduction techniques. Image deskewing (https://fsix.github.io/mnist/Deskewing.html) straightens the digits that are written in a crooked manner. Mathematically, skewing is modeled as an affine transformation: $Image^{'} = A(Image)+b$, in which the center of mass of the image is computed to estimate how much offset is needed, and the covariance matrix is estimated to approximate by how much an image is skewed. Furthermore, the center and covariance matrix are employed for the inverse affine transformation, which is referred to as deskewing. Then, we used t-SNE to reduce the dimension of the MNIST from $28\times 28$ to $2$, as the two-dimensional embedding space. 


Furthermore, we used the following four popular adversarial attack methods to evaluate the robustness of the deep learning models: (1) fast gradient method; (2) fast sign gradient method (FSGM); (3) iterative fast sign gradient method (I-FSGM); and (4) DeepFool.


## Folders ## 

**Robustness**: this directory contains codes for training deep and wide quadratic networks. The used dataset is the MNIST.<br/>

##  Running Experiments ## 

Please first go to the directory of "Robustness" .

Then, run the following code to train a wide quadratic network and a deep quadratic network, respectively.
```ruby
>> python Robustness/MNIST_QuadraticTrain_wide.py    
>> python Robustness/MNIST_QuadraticTrain_deep.py 
>> python Robustness/MNIST_QuadraticTrain_deep_RELU.py 
```
Lastly, run the following code to test the robustness of the already-trained wide and deep networks with three robustness methods.

```ruby
>> python Robustness/IFSGM_wide.py    
>> python Robustness/IFSGM_deep.py 
>> python Robustness/IFSGM_deep_relu.py 
```

```ruby
>> python Robustness/FSGM_wide.py    
>> python Robustness/FSGM_deep.py 
>> python Robustness/FSGM_deep_relu.py 
```

```ruby
>> python Robustness/DF_wide.py    
>> python Robustness/DF_deep.py 
>> python Robustness/DF_deep_relu.py 
```



