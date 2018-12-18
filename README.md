
# Deep Networks

## Introduction

The previous two labs have given you quite some insight on how neural networks work. In this lecture, you'll learn why deeper networks sometimes lead to better results, and we'll generalize what you have learned before to get your matrix dimensions right in deep networks.


## Objectives

You will be able to:

* Give intuitive justifications for using multilayer neural network designs
* Explain the terms $dZ, dW, db and da$ in the creation of a neural network

## 1. Why deep representations?

eg. computer vision:
- first layer detects edges in pictures
- second layer groups edges together and starts to detect different parts
- more layers: group even bigger parts together, etc.

eg. audio

- first layer: low lever wave features
- second layer: basic units of sounds, "phonemes" 
- third: word recognition
- fourth: sentence recognition
-...


Idea: shallow networks detect "simple" things, and the deeper you go, the more comples things can be detected. 

You can build a smaller but deeper neural network that needs exponentially less hidden units but performs better, because learning happens in each layer!

https://www.coursera.org/learn/neural-networks-deep-learning/lecture/rz9xJ/why-deep-representations

## 2. Deep network notation

### 2.1 An example


Now let's try to generalize all the notation to get things straight and know the dimensions of all matrices we'll be working with. Let's have a look at this 3-layer network:

![title](figures/Deeper_network_Day2.png)

Imagine that there are 300 cases (m = 300). What do our matrices look like? Let's start with 

$ Z^{[1]} = W^{[1]} X +b^{[1]}$

- $W^{[1]}$ is the weights matrix with dimensions (4 x 2)
- If we look at all our samples, $x$ is a (2 x 300)-matrix.
- $Z^{[1]}$ is a (4 x 300)-matrix.
- $b^{[1]}$ is a (4 x 1)-matrix. Due to broadcasting in python, this matrix will become duplicated into a (4 x 300)-matrix.

Let's take it one step further. In $ Z^{[2]} = W^{[2]} A^{[1]} +b^{[2]}$

- The dimension of $A^{[1]}$ is the same as the dimension of $Z^{[1]}$: (4 x 300)
- $W^{[2]}$ is the weights matrix with dimensions (3 x 4)
- $Z^{[2]}$ is a (3 x 300)-matrices.
- $b^{[2]}$ is a (3 x 1)-matrix. Due to broadcasting in python, this matrix will become duplicated into a (3 x 300)-matrix.


### 2.2 to generalize this all

$W^{[l]}: (n^{[l]}, n^{[l-1]})$

$b^{[l]}: (n^{[l]}, 1)$

$dW^{[l]}: (n^{[l]}, n^{[l-1]})$

$db^{[l]}: (n^{[l]}, 1)$

$ a^{[l]}, z^{[l]}: (n^{[l]}, 1)$

$ Z^{[l]}, A^{[l]}: (n^{[l]}, m)$

$ dZ^{[l]}, dA^{[l]}: (n^{[l]}, m)$


### 2.3 Forward propagation

- Input is $a^{[l-1]}$
- Output $a^{[l]}$, save $z^{[l]}, w^{[l]}, b^{[l]}, a^{[l-1]} $

#### 2.3.1 for one sample

$ z^{[l]}= W^{[l]} a^{[l-1]} + b^{[l]}$

$ a^{[l]}= g^{[l]} ( z^{[l]})$

here, $ a^{[l]}, z^{[l]}: (n^{[l]}, 1)$

#### 2.3.2 vectorized

vectorized, otherwise small z and small a (if one sample at a time), otherwise the capitals are used.

$ Z^{[l]}= W^{[l]} A^{[l-1]} + b^{[l]}$

$ A^{[l]}= g^{[l]} ( Z^{[l]})$

here, $ Z^{[l]}, A^{[l]}: (n^{[l]}, m)$

### 2.4 Backward propagation

- Input $da ^{[l]}$
- Output $da^{[l-1]}$, $dW^{[l]}, db^{[l]}$

#### 2.4.1 for one sample

$ dz^{[l]}= da ^{[l]} * g^{[l]'} ( z^{[l]})$

$ dW^{[l]} =  dz^{[l]}* a^{[l-1]T}$

$ db^{[l]} = dz^{[l]}$

$ da^{[l-1]} = W^{[l]T}*dz^{[l]}$




#### 2.4.2 vectorized

$ dZ^{[l]}= dA ^{[l]} * g^{[l]'} (Z^{[l]})$

$ dW^{[l]} = \dfrac{1}{m} dZ^{[l]}* A^{[l-1]T}$

$ db^{[l]} = \dfrac{1}{m} np.sum(dZ^{[l]}, axis=1, keepdims=True)$

$ dA^{[l-1]} = W^{[l]T}*dZ^{[l]}$


# Summary

In this brief lesson, we gave an intuitive justification behind using deep network structures and reviewed the architecture for neural nets in general. In upcoming lessons, we will begin to extend our previous work in creating a single layer neural network in order to build a deeper more powerful model.
