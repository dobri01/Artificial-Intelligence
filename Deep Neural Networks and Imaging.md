#Week10
# Online session

	just check the recording after watching the lectures 
	9:17 - 9:53


# Supervised learning

Learner receives a set of labelled training data in order to learn to predict label of unseen data.

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

# Linear regression

Learn a linear function between attributes and responses - $y=f(x)$

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