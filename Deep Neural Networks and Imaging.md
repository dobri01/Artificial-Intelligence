#Week10

# Supervised learning

The learner receives a set of labelled training data in order to learn to predict label of unseen data.

The scope is to develop a model $f(x)$ such that we are mapping the attributes $x$ to the responses $y$.  

![[Pasted image 20220425121626.png]]

 **Classification**
 - Target label $y_{n}$ is  a discrete label
 - Binary classification: $y_{n}\in \{0,1\}$ or $y_{n} \in \{-1, 1\}$
 - Multi-class classification: $y_{n} \in \{1.2. \dots , C \}$
 
![[Pasted image 20220425123745.png]]

**Regression**
- Target label $y_{n}$ is a continuous value - $y_{n}\in R$

![[Pasted image 20220425123817.png]]

Here we are not trying to represent discrete values. They are continuous values so we can't use classification.

# Differences between  regression and  classification - [source](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)
Classification is about predicting a label and regression is about predicting a quantity.
- Predictive modelling is about the problem of learning a mapping function from inputs to outputs called function approximation.
- Classification is the problem of predicting a discrete class label output for an example.
- Regression is the problem of predicting a continuous quantity output for an example.

[Predictive modeling](https://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/) is the problem of developing a model using historical data to make a prediction on new data where we do not have the answer.

Generally, we can divide all function approximation tasks into classification tasks and regression tasks.

## Classification predictive modelling

Classification predictive modelling is the task of approximating a mapping function $f$ from input variables $X$** to discrete output variables** $y$ (often called labels or categories).

It is common for classification models to predict a continuous value as the probability of a given example belonging to each output class. The probabilities can be interpreted as the likelihood or confidence of a given example belonging to each class. A predicted probability can be converted into a class value by selecting the class label that has the highest probability.

The classification accuracy is the percentage of correctly classified examples out of all predictions made.

## Regression predictive modelling
Regression predictive modelling is the task of approximating a mapping function $f$ from input variables $X$** to a continuous output variable** $y$

A continuous output variable is a real-value, such as an integer or floating point value. These are often quantities, such as amounts and sizes.

Because a regression predictive model predicts a quantity, the skill of the model must be reported as an error in those predictions. There are many ways to estimate the skill of a regression predictive model, but perhaps the most common is to calculate the root mean squared error, abbreviated by the acronym RMSE.

Some algorithms have the word “regression” in their name, such as linear regression and logistic regression, which can make things confusing because **linear regression is a regression algorithm** whereas **logistic regression is a classification algorithm**.
# Linear regression

Learn a linear function between attributes and responses: $y=f(x)$

## Univariate linear regression
Univariate means that the training data consists of samples where each sample is defined by **only one attribute value.**

 Learner model/function - maps input attributes to output responses.

**Example:**

Let's consider we can predict the target response $y=f(x)$.
- x - attribute - house size
- y = target response = house price

Training samples 
- N attribute-response pairs - $(x_{1}, y_{1}), (x_{2},y_{2}), \dots (x_{N}, y_{N})$

![[Pasted image 20220425132439.png]]

Linear function/model
- $y = mx + c$
- $y = h_{w}(x) = w_{0}+ w_{1}~x$ - where $w_{0}$ is the intercept and $w_{1}$ is the slope ($h_{w}(x)$ is the hypothesise function)

Linear function/model $h_{w}(x)$
- Learn parameters  $w_{0}$ and $w_{1}$ from past data.
- Get the training data $(x_{1}, y_{1}), (x_{2},y_{2}), \dots (x_{N}, y_{N})$ to predict $y_{new}$ for a future data $x_{new}$

![[Pasted image 20220425133421.png]]

### What is a good model?
- Generate a line that passes "as close as possible" to the past example points. 
- Loss function: loss between 'ground truth' y and model prediction $h_{w}(x)$

$$Loss(h_{w}) = \sum\limits_{j=1}^{N}(y_{j}-h_{w}(x_{j}))^{2}= \sum\limits_{j=1}^{N}(y_{j}- (w_{0}+ w_{1}x_{j}))^{2}$$

How do we find the best $w_{0}$ and $w_{1}$ (minimise least squares error)?
- Finding these weights is the function modelling (or learning).
$$w = argminLoss(h_{w})= argmin \sum\limits_{j=1}^{N}(y_{i}-(w_{0}+w_{1}x_{j}))^{2}$$

This is a minimisation problem. 
- **Closed form solution** - analytical solution - when partial derivatives (of the loss function) with respect to $w_{0}$ and $w_{1}$ are zero.
$$\frac{\partial}{\partial w_{0}} \sum\limits_{j=1}^{N}(y_{j}-(w_{0}+w_{1}x_{j}))^{2} = 0 $$
$$\frac{\partial}{\partial w_{1}} \sum\limits_{j=1}^{N}(y_{j}-(w_{0}+w_{1}x_{j}))^{2} = 0 $$

From this equation we get a unique solution:
$$w_{0}= \frac{1}{N}(\sum\limits y_{j} - w_{1}(\sum\limits x_{j}) )$$
$$w_{1}= \frac{N\left(\sum\limits x_{j}y_{j}\right)- (\sum\limits x_{j})(\sum\limits y_{j})}{N(\sum\limits x_{j}^{2})- (\sum\limits x_{j})^{2}}$$

Using these we can determine the parameters only using the training data.

- **Search optimisation** - numerical solution - hill-climbing - follow the gradient of the function to be optimised

Gradient descent:
- Choose a starting point in the weight space
- Move to a downhill neighbouring point
- Repeat until convergence

Gradient descent algorithm:

![[Pasted image 20220425221230.png]]

![[Pasted image 20220425230746.png]]

Batch gradient descent (i.e. modify weights on the basis of all N training examples) - for the whole dataset, usually you take the average but it is adjusted in the $\alpha$ parameter (learning rate).
$$w_{0}= w_{0}+ \alpha \sum\limits_{j}(y_{j}-h_{w}(x_{j}))$$ 
$$w_{1}= w_{1}+ \alpha \sum\limits_{j}(y_{j}-h_{w}(x_{j}))$$ 
The convergence is guaranteed as long as we pick a small enough $\alpha$.

If $\alpha$ is **too small** then the convergence is very slow and we might terminate the algorithm before getting to it.
If $\alpha$ is **too big** it might be jumping too much and not getting the optimal parameters.

## Multivariate linear regression
Multivariate means that the training data consists of samples where each sample is defined by **more then one attribute value.**

Every example $x_{j}$ is an $n$-element vector.

Our function model:
$$h_{w}(x_{j}) = w_{0}+ w_{1}x_{j,1}+ \dots + w_{n}x_{j,n}= w_{0}+ \sum\limits_{i=1}^{n}w_{i}x_{j,i}$$
$$h_{w}(x_{j}) = w_{0}x_{j,0}+ \sum\limits_{i=1}^{n}w_{i}x_{j,i}$$
	$$h_{w}(x_{j}) = \sum\limits_{i=0}^{n}w_{i}x_{j,i}= w^{T}x_{j} ~~~~~~ \text{ where } x_{j,0}=1$$

Batch gradient descent:

$$ w_i = w_{i} + \alpha \sum\limits_{j}x_{j,i}(y_{i}- h_{w}(x_{j}))$$

![[Pasted image 20220425234106.png]]

# Linear classification

Find a linear function hypothesis $h_{w}(x)$ (linear decision boundary) that separates the two classes given this linearly separable data.

![[Pasted image 20220427003513.png]]
![[Pasted image 20220427005842.png]]

How are the weights changed and what's their interpretation?

$h_{w}$ is represented by the threshold function which has undefined gradient at 0 and  0 gradient elsewhere. So analytical and numerical solutions are not available.

How many times do we update the weights?
- One example at a time, run through the entire set of training examples.
- This is typically called an **[[Definitions#Epoch|epoch]]**.

![[Pasted image 20220428095434.png]]
The model is too simple so the performance is not that stable.

## Perceptron learning rule - updating the weights
$$w_{i}= w_{i}+ \alpha x_{i}(y- h_{w}(x))$$
- $w_{i}$ - the old weight.
- $\alpha$ - learning rate
- $x_{i}$ - corresponding attribute
- $y-h_{w} (x)$ - difference between the **actual label** and the **predicted label**.

*Case 1:* What if the output is correct? - $y=h_w(x)$
- No change in weights


*Case 2:* What if $y=1$ but $h_{w}(x) =0$?
 The model is predicting values smaller then it should so we need to make the $w^{T}x$ bigger so that $h_{w}(x)$ outputs a 1.
 - $w_{i}$ is increased when the corresponding input $x_{i}$ is positive.
 - $w_{i}$ is decreased when the corresponding input $x_{i}$ is negative.

*Case 3:* What if $y=0$ but $h_{w}(x) =1$?
The model is predicting values bigger then it should so we need to make $w^{T}x$ smaller so that $h_{w}(x)$ outputs a 0.
 - $w_{i}$ is decreased when the corresponding input $x_{i}$ is positive.
 - $w_{i}$ is increased when the corresponding input $x_{i}$ is negative.
## Problems with linear classification
- It has a hard threshold - either 0 or 1 (might not be the case)
- The function isn't differentiable so weights learning could be unpredictable (it's not guided by the gradient). 
- The model performance could fluctuate (unstable) - the model is not mathematical based, it is based on intuition.  





# Logistic regression

It is essentially a binary classification algorithm in machine learning. It can also be used as multi-class classification.

Logistic regression replaces the threshold function with the logistic function.
$$h_{w}(x) = Logistic(w^{T}x)$$
This has a few benefits:
- It now has a soft threshold
- We can consider the logistic value as an estimate of confidence 

![[Pasted image 20220428162923.png]]

It uses a regression like function modelling approach to perform binary classification - the values are continuous but are in the range $[0,1]$. We can use those values as probability estimates.
$g$ - logistic function
$w$ - the parameters
$x$ - the attributes 

- For 1D: $x=[1~x], w=[w_{0}~ w_{1}]$
- For 2D:

![[Drawing 2022-04-28 16.57.01.excalidraw| 250]]

The logistic function provides a binary classification with a probability estimate.

![[Pasted image 20220428170811.png]]

Example:

$h_{w}(x) = g(w^{T}x) = \frac{1}{1+e^{-w^{T}x}}$
$h_{w}(x)$ can be considered as probability estimate.

$h_{w}(x) = p(y=1 | x; w)$ - probability that $y=1$, given attributes $x$ and function parameterised by $w$
$h_{w}(x) = p(y=0 | x; w)$ - probability that $y=0$, given attributes $x$ and function parameterised by $w$

$p(y=1 | x; w) +p(y=0 | x; w) = 1$

Let's recall $g(z)$ is thresholded to perform binary classification.
- $h_{w}(x)=g(z) \geq 0.5$ - assign class label $1$, i.e. $w^{T}x \geq 0$
- $h_{w}(x)=g(z) < 0.5$ - assign class label  $0$, i.e. $w^{T}x < 0$


## How do we find the parameters w?
Classifier (logistic regression) learns parameters $w$ from training data $x_{1}, x_{2}, \dots , x_{N}$ (attributes) and $y_{1}, y_{2}, \dots , y_{N}$ (targets) so that it can later classify $x_{new}$

**To make a prediction for a new sample $x_{new}$:**
$$y_{new}= h_{w}(x_{new}) = g(w^{T}x_{new}) = \frac{1}{1 + e^{-w^{T}x_{new}}}$$

*Loss function:*
$$Loss(w) = (y-h_{w}(x))^{2}$$
- $y$ - actual label
- $h_{w}(x)$ - model predicted label

Learning means finding the parameters $w$ that minimise the loss: **$argmin ~ Loss(w)$**

*To find the new $w$:*
$$w_{i}= w_{i} - \alpha \frac{\partial}{\partial w_{i}}Loss(w)$$
- $w_{i}$ - the new, the old
- $\alpha$ - learning rate
- $\frac{\partial}{\partial w_{i}}Loss(w)$ - learning rate, the derivative of $Loss$ function with respect to $w_{i}$.

Repeat for all $w_{i}$ components of $w$.

Can be applied to as many dimensions as needed (as long as it is linearly separable).

**Loss function for arbitrary training example (x,y):**

$$\frac{\partial}{\partial w_{i}}Loss(w) = \frac{\partial}{\partial w_{i}}(y-g(w^{T}x))^{2}$$

after some calculation ...
**$$\frac{\partial}{\partial w_{i}}Loss(w) = -2(y-h_{w}(x)) \times h_{w}(x) \times (1-h_{w}(x)) \times x_{i}$$**


**To find the new $w$**, we plug the loss function to what we had earlier:
**$$w_{i} = w_{i}+ \alpha (y-h_{w}(x)) \times h_{w}(x) \times (1-h_{w}(x)) \times x_{i}$$**

1) The output is correct - $y=h_{w}(x)$ 
	- no change in weights
1) $y=1$ but $h_{w}(x) = 0$

	- make $w^{T}x$ bigger so that $h_{w}(x)$ outputs a value closer to $1$
	- $w_{i}$ is increased when the corresponding input $x_{i}$ is positive
	- $w_{i}$ is decreased when the corresponding input $x_{i}$ is negative
3. $y=0$ but $h_{w}(x) = 1$
	- make $w^{T}x$ smaller so that $h_{w}(x)$ outputs a value closer to $0$
	- $w_{i}$ is decreased when the corresponding input $x_{i}$ is positive
	- $w_{i}$ is increased when the corresponding input $x_{i}$ is negative

**How many times do we update the weights?**
- One example at a time, run through the entire set of training examples.
- This is typically called an **[[Definitions#Epoch|epoch]]**.
- Repeat until converged (not all training samples are correctly labelled) or until some stop criteria (a set number of epochs).

![[Pasted image 20220428182010.png]]

Much more stable then [[Deep Neural Networks and Imaging#Linear regression|Linear regression]].

# Neural networks

## Neuron
![[Pasted image 20220510170614.png]] 

A neuron 'fires' when a linear combinations of its inputs exceeds some threshold (hard or soft).
- $a_{i}$ : data input component from $i^{th}$ neuron. (we used to use the notation $x$ for attributes but for neural networks we use $a$)
- $w_{i,j}$ : connection weight from $i^{th}$ neuron to $j^{th}$ neuron.
- $i~n_{j}= \sum\limits_{i=0}^{m}w_{i,j}~a_{i}$ : weighted linear combination of inputs at $j^{th}$ neuron.
- $g(z)$ : activation function. (e.g. threshold, sigmoid)
- $a_{j}= g(\sum\limits_{i=0}^{m}w_{i,j}~a_{i})$ : activation value/output of $j^{th}$ neuron.
### Thresholds 
 **Perceptron** - similar to [[#Linear classification| linear classification]]
 - Hard threshold
 - $g(w^{T}x)=1$ , if $w^{T}x >0$
 - $g(w^{T}x)=0, otherwise$

![[Pasted image 20220510171946.png |150]]

**Sigmoid perceptron** - logistic function (similar to [[#Logistic regression| logistic regression]])
- Soft threshold
- $q(w^{T}x)= \frac{1}{1+e^{-w^{T}x}}$
- The weights determine the location and slope of the sigmoid

![[Pasted image 20220510172019.png|150]]

## Types of Neural Networks
![[Pasted image 20220510172514.png]]

### Perceptron
**Learning rules :**
- Perceptron learning rule: 
	- $w_{i}= w_{i}+ \alpha x_{i}(y-h_{w}(x))$
- Gradient descent rule (e.g. for sigmoid perceptron)
	- $w_{i}= w_{i}- \alpha \frac{\partial}{\partial w_{i}}Loss (w)$
	- $w_{i}= w_{i}+ \alpha (y-h_{w}(x)) \times h_{w(x)}\times (1-h_{w}(x))\times x_{i}$

These are the same rules as used in linear classification and logistic regression earlier.

$w$ is referring to the parameters which are on the edges between the neural connections. 

Both $x$ and $a$ are attributes and might be used interchangeably. Also $x_{0}=1$ or $a_{0}=1$ that is considered the bias.

**Perceptron network - single-layer FF NN**
- Train a network to add two binary inputs
- A 2 outputs network is considered as 2 separate networks, weights feeding in  output 1 don't contribute to output 2
	- we can use perceptron or gradient descent rule on each output unit separately

![[Pasted image 20220510182359.png]]

**Linear separability**

We can see that in this one layer neural network the data is not linearly separable.

![[Pasted image 20220510182733.png]]

### Multi-layer Feed Forward Neural networks -ML  FF NN

A neural network with one or more hidden layers can represent a non-linear decision function.
It may need many neurons in the hidden layer.

![[Pasted image 20220510183347.png]]


![[Pasted image 20220510200056.png]]

How do we train this ~~beast~~ network?

![[Pasted image 20220510200421.png]]

**Two parts:**
- weights update for output layer
- weights update for hidden layer

#### Back propagation algorithm
1. Update weights at the output units, using the observed error
2. Starting back from the output layer, repeat successively for each hidden layer until the first hidden layer
	- Propagate the weight update proportion back to the previous layer
	- Update the weights to the previous layers


![[Pasted image 20220510201908.png]] 
![[Pasted image 20220510203254.png]]

Now we can update the weights in the hidden layer 

![[Pasted image 20220511123920.png]]
![[Pasted image 20220511124032.png]]

Back propagation algorithm:
1. Update weights at the output units, using the observed error $w_{i,k} = w_{j,k} + \alpha a_{j} \Delta_{k}$
2. Starting back from the output layer, repeat successfully for each hidden layer until the first hidden layer.
	- Propagate the weight update proportions back to the previous layer
	- Update the weights between the two layers

# Deep neural networks 

## Training errors - underfitting, overfitting

**Underfitting** - a phenomenon where the model is doing worst then expected on the training data.

**Overfitting** - a phenomenon where the model is doing a perfect job on the training data but it results in a complex boundary that will not perform good in practice due to edge cases.

**Generalisation** - how well the model performs on unseen data. Generalising well from the training data to the testing data.

![[Pasted image 20220511130056.png]]

## Regularisation

Regularisation is a common method for achieving generalisation in machine learning.

**Regularisation loss**: $Loss^{req}(w) = Loss(w)+ \lambda RegTerm$
- $Loss(w)=(y-h_{w}(x))^{2}$
- Regularisation parameter/rate $\lambda$ - between 0 and 1, the larger the higher the proportion or regularisation.


**$L_{2}$ regularisation** - magnitude of the weight parameters 
- $Loss^{req}(w)=Loss(w ) + \lambda ||w||^{2}_{2}$
- $Loss^{req}(w)= Loss (w) + \lambda \sum\limits_{w_{i}} w_{i}^{2}$

Recall the old  weight update rule: $w_{i}= w_{i}- \alpha \frac{\partial}{\partial w_{i}}Loss (w)$

We get out weight update rule with regularised loss:
$$w_{i}= w_{i}- \alpha [-2(y-h_{w}(x)) \times h_{w}(x) \times (1- h_{w}(x)) \times x_{i}+ \lambda w_{i}]$$

## Drop out
Another common method for achieving generalisation in deep neural networks. Drop out encourages the network to not memorise the learning but rather adapt.

$\pi$ = the probability of a weight becoming 0.

![[Pasted image 20220514142114.png]]

## Stochastic gradient descent

For example, distribute 10000 training samples in 10 mini-batches
- Each subset is 1000 samples
- We'll update the weights at the end of each iteration
- In this instance, one epoch consists of 10 iterations
- In deep neural networks, one often needs thousands of epochs to learn

Benefits
- Quicker to converge in practice.
- Helps to avoid overfitting, as at each iteration we're only learning from a subset of the entire data.
- Achieves generalisation faster.


## Activation function

**Logistic function**
- $g(w^{T}x)=\frac{1}{1+e^{-w^{T}x}}$

**Rectified linear unit - ReLU**
- $g(w^{T} x) =max(0, w^{T}x)$
- A commonly used activation function in deep NN due to its useful properties.
	- Computational efficiency (easier to compute then the logistic function).
	- Sparse activation avoids overfitting.



![[Pasted image 20220514144854.png|200]]


 # Convolutional neural network

Convolutional neural networks is a neural network formulation and is mostly consists of:
- [[Deep Neural Networks and Imaging#Neuron|Neurons,]]
- [[Deep Neural Networks and Imaging#Back propagation algorithm|Back propagation,]]
- [[Deep Neural Networks and Imaging#Convolutional neural network|Convolution neural network,]]
- [[Deep Neural Networks and Imaging#Drop out|Drop out,]]
- [[Deep Neural Networks and Imaging#Stochastic gradient descent|Stochastic gradient descent]]

Convolutional neural networks are inspired from the physiology of our visual cortex where a **receptive field** (filter/kernel) helps in a visual understanding. In NN context, let's look at a 4-unit **receptive field.**

The receptive field gets data and feeds it to a neuron which can process it through an activation function. 

![[Pasted image 20220514145216.png]]  

Consider a 5x5 image which is flattened from 2D to 1D and passed through a convolution layer. 

If we have a 4-units receptive field on a 1D convolution layer then we'll get 7 outputs from 10 samples. (neurons 1-4 give one output, neurons 2-5 give the second output)

**Padding** can be used to keep the output the same number as the input. It can be done both on the left and on the right. (same for multi dimensions)

If we don't use **padding** we will not have the same sample but the result is valid.

The neurons are the receptive units and the weights are the connections. (blue balls)

A **stride** is how many steps we move the receptive field. In our example we move from 1-4 to 2-5 so the stride is 1.

![[Pasted image 20220514173602.png]]

### 2D Convolutional layer

When we design a convolutional layer we have to take into account these things:
- receptive field size (filter/kernel)
- stride size
- input size
- output size

![[Pasted image 20220514175115.png]]

Example

We do have a formula as well for it:
$$\frac{w-k+2p}{s}+1$$
where:
- w - input size (width)
- k - kernel size / filter size / receptive field size
- p - padding 
- s - stride size

![[Pasted image 20220514181006.png]]

### Feature maps

Consider a 2D convolution layer, an input image of size 10x10 and a receptive field of size 3x3. The output will be 8x8.

Consider a colour image, which is made up of 3 channels (RGB) so 10x10x3. 
Typically our receptive field will also need 3 channels so 3x3x3.
The output will be 8x8. (with a "valid" output i.e. no padding)

Consider now multiple receptive fields, 5 for example each of size 3x3x3 and a 10x10x3 image. Then the output will be 8x8x5. (with a "valid" output i.e. no padding)
We call each output image from these 5 a **feature map** which *highlights* a particular *feature* from the image. Think of each receptive field as a filter.

![[Pasted image 20220514182037.png]]

### Polling layer

Each feature map needs to be passed through a pooling layer.

Note that multiple receptive fields generate many feature maps. More and more convolution layers will enhance feature representation. We need to *compactify* the deeper representation.

Pooling layer
- max-pooling
- average-pooling
- min-pooling


Here is an example of Max-pooling (picking the largest)

![[Pasted image 20220514190243.png]]


### Fully connected layers
When it gets to this stage, the image is flatten in a single vector and it connects it to the output layer.

Here is a illustration with all the layers:

![[Pasted image 20220514191955.png]]

For binary classification we need only one unit in the output layer.

For multi-class classification, it's convenient to put as many units as the number of classes using a **one-hot encoding system**.

#### One-hot encoding
![[Pasted image 20220514192333.png|200]]

At the end of the figure you can notice $\sigma$, that is the **activation function**.

#### SOFTMAX function
The softmax function is an extension of the logistic function for multiple classes.
$$\huge \sigma(a_{k}) = \frac{e^{a_{k}}}{\sum\limits_{j=1}^{O}e^{a_{j}}}$$

- $a_{j}: j^{th}$ unit's activation from the output layer
- $a_{k}:k^{th}$ unit's activation from the output layer
- $O:$ the number of units in the output layer

The softmax function transforms the output activation values to a range of 0 to 1, such that they all add up to 1. These values can be interpreted as a probability of classification for each class.

## Hyper-parameters

These are parameters are not learnable but are manually set by hit and trial or research.
- number of convolution layers
- number of pooling layers
- what pooling function - e,g, average, max
- number of receptive fields (filters)
- receptive field size
- number of fully connected layers
- where to put a layer - pooling, convolutional
-  what learning rate
- what optimisation function - stochastic gradient descent, adam
- what loss function to use - sum of squared differences, cross-entropy
- what activation function - ReLU (Rectified linear unit), tanh, logistic


# Deep neural networks and imaging

 A digital image is usually  constructed as a function of illuminance and reflectance.
 - $f(x,y)=i(x,y)~r(x,y)$
	 - $0<i(x,y)<\infty$
	 - $0<r(x,y)<\infty$

## Image representation
An image $f(x,y)$ - equally spaced samples arranged as ($M \times N$) array. (matrix with origin at the top left, N for columns and M for rows).

Space wise it takes $M \times N \times k$ bits to store an image.

**Binary image** 
- $f(x,y) \in \{0,1\}$

**Gray-scale image** 
- $f(x,y \in C)$ 
- $C=\{0, \dots , 255\}$

**Colour image**
- $f_{R}(x,y) \in C$, $f_{G}(x,y) \in C$, $f_{B}(x,y) \in C$
- $C=\{0, \dots , 255\}$ for a 24 bits image

![[Pasted image 20220515112947.png]]

 ## Generative adversarial networks - GAN
 
![[Pasted image 20220515114346.png]]


# Exercises

  ![[Pasted image 20220516155400.png]]


![[Pasted image 20220516155417.png]]
![[Pasted image 20220516155433.png]]

![[Pasted image 20220516155454.png]]
![[Pasted image 20220516155717.png]]

![[Pasted image 20220516155507.png]]