# Week 4 - Joint Distribution and statistical inference
## Joint Distributions
Conceptually, joint distributions are multi-dimensional [[Probability#Probability Density Function - PDF|PDFs]] (or [[Probability#Probability Mass Function - PMF|PMFs]] or [[Probability#Cumulative Distribution Function - CDF|CDFs]])
$$f_{X_1 ... X_N}(x_{1}...x_{N})$$
### Joint PMF
Let $X$ and $Y$ be two [[Probability#Discrete Random variables|discrete random variables]]. **The joint PMF** of $X$ and $Y$ is defined as
$$P_{X,Y}(x,y) = P(X = x~~and ~~Y = y)$$
According to the definition, we know that joint [[Probability#Probability Mass Function - PMF|PMFs]] indicates the probability of the intersection of events {${\xi : X(\xi) = x}$} and {${\xi : Y(\xi)=y}$}. Notice this definition requires both $X$ and $Y$ to be  [[Probability#Discrete Random variables|discrete random variables]].

![[Pasted image 20220312191941.png]]
<center>A joint PMF for a pair of discrete random variables consists of an array of impulses. To measure the size of the event A, we sum all the impulses inside A </center>

### Joint PDF
Let $X$ and $Y$ be two [[Probability#Continuous Random Variables|continuous random variables]]. **The joint PDF** of $X$ and $Y$ is a function $f_{X,Y}(x,y)$ that can be integrated to yield a probability
$$P(A)= \int_{A} f_{X,Y}(x,y)~dx~dy$$
The joint PDF can be interpreted as the probability per unit, and can take values larger than 1.
 
![[Pasted image 20220312193758.png]]

### Marginal PMF and Marginal PDF
**The marginal PMF **is defined as 
$$P_{x}(x)= \sum\limits_{y \in Y(\Omega)} P_{X,Y}(x,y) ~~~ and ~~~ P_{Y}(y) = \sum\limits_{x \in X(\Omega)} P_{X,Y}(x,y)$$  

**The marginal PDF** is defined as
$$f_{X}(x) = \int _{Y(\Omega)}f_{X,Y}(x,y) dy~~~ and ~~~ f_{Y}(y) = \int _{X(\Omega)} f_{X,Y}(x,y) dx$$

### Joint CDF

Let $X$ and $Y$ are **discrete**, then
$$\huge F_{X,Y}(x,y) = \sum\limits_{y'\leq y}\sum\limits_{x'\leq x} P_{X,Y}(x'y')$$

Lex $X$ and $Y$ are **continuous**, then
$$F_{X,Y}(x,y)= \int_{-\infty}^{y}\int_{-\infty}^{x}f_{X,Y}(x',y') dx'dy'$$

The dots represent the pairs $(x_i ,y_{i})\in X(\Omega) \times Y(\Omega)$. $F_{X,Y}(x,y)$ is the probability that $(X,Y)$ belongs to the shaded region.
![[Pasted image 20220313142501.png]]
## Independency
We say that two random variables $X$ and $Y$ are independent if
$$F_{X,y}(x,y) = F_{x}(x)F_{y}(y)$$
Intuitively we say two random variables are **independent** iff the joint [[Probability#Cumulative Distribution Function - CDF|CDF]] can be factorised into [[Probability#Cumulative Distribution Function - CDF|CDFs]] of [[Probability#Discrete Random variables|single random variables]]. This factorisation shows no correlation between these two [[Probability#Discrete Random variables|single random variables]].

### Independence for two variables
We say two [[Probability#Discrete Random variables|discrete random variables]] $X$ and $Y$ are **independent** iff
$$P_{X,Y}(x,y) = P_{X}(x) P_{Y}(y)$$
We say two [[Probability#Continuous Random Variables|continuous random variables]] $X$ and $Y$ are **independent** iff
$$f_{X,Y}(x,y)=f_{X}(x) f_{Y}(y)$$
### Independence for multiple variables
We say a sequence of random variables $X_{1}, X_{2}, \dots, X_{N}$ are independent iff the [[Statistics#Joint PDF|joint PDF]] can be factorised
$$f_{X_{1},\dots, X_{N}}(x_{1}, \dots ,x_{N)}= \prod_{n=1}^{N}f_{X_{n}}(x_n)$$

## Conditional PMF and Conditional PDF
### Conditional PMF
We say that for a [[Probability#Discrete Random variables| discrete random variable]] $X$ and event $A$, **the conditional PMF** of $X$ given $A$ is defined as
$$
\begin{align}
P_{X|A}(x_{i}) = P(X= x_{i}|A) \\
= \frac{P(X = x_{i}~and~A)}{P(A)}, && for ~any ~x_{i}\in X(\Omega)
\end{align}
$$

Let $X$ and $Y$ be two [[Probability#Discrete Random variables| discrete random variables]]. **The conditional PMF** of $X$ given $Y$ is
$$P_{X|Y}(x|y) = \frac{P_{X,Y}(x,y)}{P_{Y}(y)}$$
According to the definition, *the conditional PMF* is the division of [[Statistics#Joint PMF|the joint PMF]] and [[Statistics#Marginal PMF and Marginal PDF|the marginal PMF]].

![[Pasted image 20220313144832.png]]

### Conditional PDF
Let $X$ and $Y$ be two [[Probability#Continuous Random Variables|continuous random variables]]. The conditional PDF of $X$ given $Y$ is
$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_{Y}(y)}$$
**The conditional PMF** can be interpreted as the [[Probability#Conditional Probability|conditional probability]] per unit. Therefore, **the conditional PDF** can take values larger than 1. This is consistent with the fact that PDF can be interpreted as the probability per unit.

## Statistical Inference

### Descriptive Statistics
 Descriptive Statistics refers to a summary that quantitatively describes features from a collection of information, which does not assume that the data comes from a larger population.
 - **Central tendency**: expectation (mean), [[Definitions#Median|median]], mode. 
 - **Dispersion**: the range and [[Definitions#Quartile|quartiles]] of the dataset
- **Spread**: variance and standard deviation. 
- **Shape of the distribution**: [[Definitions#Skewness|skewness]] and [[Definitions#Kurtosis|kurtosi]]

### Expectation
The expectation of a random variable $X$ is
$$E[X] = \sum\limits_{x\in X(\Omega)} x P_{X}(x)$$
Where:
- $x\in X(\Omega)$ is the sum over all states
-  $x$ = the state X takes
- $P_{X}(x)$ is the percentage

The expectation gives the value we expect for a [[Probability#Discrete Random variables|  random variables]] before implementing an experiment.


#### Linearity of expectation
$$E[X] = E[X_{1}+X_{2}+X_{3}] = E[X_{1}]+ E[X_{2}]+ E[X_{3}]$$

#### Sample Mean and Expectation
Suppose we repeatedly do the same random experiment. This leads to n random variables $X_{i}, ~i\in [n]$ with the same [[Probability#Probability Mass Function - PMF|PMF]].
We can define the **Sample mean** $\bar{X}_{n}$ as
$$\bar{X}_{n} = \frac{1}{n} \sum\limits_{i=1}^{n}X_{i}$$
##### The difference between Sample mean and Expectation
**Expectation** $E[X]$
- A statistical property of a [[Probability#Discrete Random variables|random variable]].
- A deterministic number independent of implementation of the experiment. 
- Often unknown, or is the center question of estimation. It is a quantity about the population.

**Sample Mean** $\bar{X}_{n}$
- A numerical value. Calculated from data
- Itself is a [[Probability#Discrete Random variables|random variable]]. Note is is an average of $X_{i}$, each of which is a [[Probability#Discrete Random variables|random variable]].
- It had uncertainty. The uncertainty reduces as more samples are used. The reason is that the randomisation is likely to offset each other with more experiments.
- We can use sample mean to estimate the expectation: the expectation cannot be computed from the data, while the sample mean can.
### Variance and standard deviation
Variance, $Var(X)$ is the expected value of the squared derivations
$$Var(X) = E[\triangle^{2}] =E[(X-\mu)^{2}] = E[(X-E[X])^{2}]$$

The **Standard Deviation** - $\sigma$ is the square-root of the variance:
$$\sigma = \root \of {E[\triangle ^2]}$$
If the variance is large, then this means that the derivation $\triangle$ can be large with high probability. If the variance is small, then the deviation $\triangle$ would be small with high probability.

**Therefore, the variance measures the uncertainty of a random variable.**

**Variance is a measure of risk**: if the variance $X$ is larger it has more uncertainty. 

#### Proprieties of Variance
Let $X$ be a random variable. Then we have the following property
1. Variance is the expectation of square minus the square of expectation
$$Var(X) = E[X^{2}]- (E[X])^{2}$$
2. Scale. For any constant c
$$Var(cX) = c^{2}Var(X)$$
3. Shift. For any constant c
$$Var(X+c) = Var(X)$$
4. If $X$ and $Y$ are [[Probability#Independence|independent]] then
$$Var(X+Y)= Var(X) + Var(Y)$$


### Continuous Random Variables
#### Expectation for continuous variables 
For a continuous random variable $X$ with the [[Probability#Probability Density Function - PDF|PDF]] $f_{X}(x)$, the expectation is defined as
$$E[X]=\int _{-\infty} ^{\infty}xf_X(x)dx$$
#### Variance for continuous random variables
For a continuous random variable $X$ with the [[Probability#Probability Density Function - PDF|PDF]] $f_{X}(x)$ and the expectation $E[X]=\mu _X$, the [[Statistics#Variance and standard deviation|variance]] is defined as
$$Var(X)=E[(X-\mu_X)^2]=\int _{-\infty} ^{\infty}f_X(x)(x-\mu_{X})^{2}dx $$
Similar to the case with discrete random variables, the expectation for [[Probability#Continuous Random Variables|continuous random variables]] is a quantity we expect before the implementation of the experiment, while the [[Statistics#Variance and standard deviation|variance]] **measures the concentration behaviour of a random variable around its expectation**.

### Statistical Inference
Statistical inference refers to the process that collects data to estimate the desired quantity about the [[Definitions#Population|population]].

#### Point estimation
Point estimation is a process that samples data to calculate a single value as a "best estimate" of an unknown [[Definitions#Population|population]] parameter.

##### Point estimation of expectation
After collecting data using [[Statistics#Random sampling| sampling with replacement]] we estimate the average height by the average height estimator as
$$\hat{\Theta}=\bar{X} = \frac{X_{1}+X_{2}+ \dots + X_n}{n}$$
with: 
- **random samples**: $X_{1}+X_{2}+ \dots + X_n$ are [[Probability#Independence|independent]] and identically distributed [[Probability#Continuous Random Variables|continuous random variables]], which have the same distributions:
$$F_{X_{1}}(x)=F_{X_{2}}(x)= \dots = F_{X_{n}}(x)$$
- **point estimator**: The average height estimator $\hat{\theta}$ is a random variable called a point estimator.

#### Point estimator 
Suppose we have a random sample $X_{i}$ with the same distribution $i=1, \dots , n$. A point estimator $\hat{\Theta}$ for an unknown parameter $\theta$ is defined as a function of the random sample:
$$\hat{\Theta} = h(X_{1}, \dots , X_{n})$$
where h can be any function
#### Estimate Estimator differences
- An **estimate** is a number. It is the random realisation of a random variable. This value can be computed from the empirical data.
- An **estimator** is a [[Probability#Discrete Random variables|random variable]]. It takes a set of [[Probability#Discrete Random variables|random variables]] as inputs and generates another [[Probability#Discrete Random variables|random variable]]. It gives a formula to estimate the parameter. *We can view the estimate as a realisation of the estimator*
#### Random sampling
We choose a random sample of size n with replacement from the population.
- We choose n persons uniformly and independently from the population 
- Sampling **with** replacement: we allow one person to be chosen twice
- Sampling **without** replacement: we don't allow one person to be chosen twice

**Why use sampling with replacement**: Random variables $X_i$ are independent which simplifies the analysis.

**When to use sampling with replacement**: when the population is large enough that the probability of choosing one person twice is extremely low.

## Bias 
Let $\hat{\Theta}=h(X_{1}, \dots , X_{n})$ be a point estimator for for $\theta$. **The bias** of the [[Statistics#Point estimator|point estimator]] $\hat{\Theta}$ is
$$B(\hat{\Theta}) = E[\hat{\Theta}] - \theta$$
An estimator $\hat{\Theta}$ is unbiased iff $B(\hat{\Theta}) = 0$

We can compute the deviation of the estimator from $\theta$, which itself is a random variable. The mean squared error is the expected square of this deviation.

## Mean Squared Error (MSE)

The MSE of a [[Statistics#Point estimator|point estimator]] $\hat{\Theta}$, denoted as $MSE(\hat{\Theta})$, is defined as
$$MSE(\hat{\Theta}) :=E[(\hat{\Theta}-\theta)^{2}] = Var(\hat{\Theta})+B(\hat{\Theta})^{2}$$

## Consistency 
Let $\hat{\Theta} _{2},~i= 1, \dots , n , \dots$ be a  sequence of points estimators of $\theta$. We say that $\hat{\Theta}$ is a consistent estimator of $\theta$ if
$$lim_{n\rightarrow \infty}~P(|\hat{\Theta} - \theta| \geq \epsilon) = 0, ~~~ for~all~\epsilon>0$$
Below is an illustration of the probability of error $P(|\hat{\Theta}_{n} -0| \geq \epsilon)$ with $\epsilon=1$ for an estimator $\hat{\Theta}_{n}$. The shaded area corresponds to the event $|\hat{\Theta}_{n} -0| \geq \epsilon$. As n grows, we see that the probability of error diminishes. This shows that this estimator is consistent.

![[Pasted image 20220314145259.png]]

# Week 5 - Maximum Likelihood Estimation and Logistic Regression 

## Maximum Likelihood Estimation
Informally, the **Maximum likelihood estimation** searches the best parameters of a probability distribution that makes the data most likely.
### Likelihood function
Let $X = X_{1}, X_{2}, \dots X_{n}$ be a random sample from a distribution with a parameter $\theta$. Suppose we have a vector $x =(x_{1},x_{2}, \dots x_{n})$ , then 
- if $X_{i}$'s are [[Probability#Discrete Random variables|discrete]]:
$$L(\theta|x)=L(\theta |x_{1},x_{2}, \dots , x_{n}) = P_{X}(x;\theta)$$
where $P_{X}(x;\theta)$ is the [[Probability#Probability Mass Function - PMF|PMF]] of $X$ parameterised by $\theta$

- if $X_{i}$'s are [[Probability#Continuous Random Variables|continuous]]:
$$L(\theta | x) = L(\theta| x_{1}, x_{2}, \dots , x_{n}) = f_X(\theta|x)$$

### Probability and Likelihood differences

**Probability**: a number $p \in [0,1]$ between 0 and 1 to describe how likely an event is to occur, or how likely it is that a proposition is true, assuming we know the distribution of the data.

**Likelihood**: a function that measures the goodness of fit of a statistical model to a sample of data for given values of the unknown parameters. It is a function of the unknown parameters, e.g. $\theta$

### Maximum Likelihood Estimator - MLE
Let $X = (X_{1},X_{2}, \dots , X_{n})$ be a random sample from a distribution with a parameter $\theta$. Suppose we have a vector $x =(x_{1},x_{2}, \dots x_{n})$, a maximum likelihood estimate of $\theta$ denoted as $$\hat{\theta}_{MLE}=argmax ~L(\theta|D)$$

A maximum likelihood estimator of the parameter $\theta$, denoted as $\hat{\Theta}_{MLE}$ is a random variable $\hat{\Theta}_{MLE} = \hat{\Theta}_{MLE}(X)$ whose value when $X_{1}= x_{1}, \dots , X_{n}= x_{n}$ is given by $\hat{\theta}_{MLE}$

#### How to solve this?
- **exhaustive search**
	- grid search - usually used for tuning hyper-parameters of a machine learning model. ([check this](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search|))
	
- **optimisation algorithms** - a more general way to solve the problem.

### Cost functions
 **The cost function** is a function that maps a set of events into a number that represents the "cost" of that event occurring. Also known as the **loss function** / **objective function**

**The cost function for likelihood** is a general one-to-one mapping with likelihood - the negative logarithm of the likelihood function:
$$J(\theta,D)= -log(L(\theta|D))$$

### Optimisation
Optimisation means finding the best solution from among the set of all feasible solutions.

**Optimisation procedure**
- Constructing a model
- Determining the Problem Type
- Selecting an optimisation algorithm

**Some Machine Learning examples**:
- *Supervised learning*: Given some training data, we want to train a machine learning model to explain the data. The training process is essentially a process of finding a optimal set of parameters of this model and the optimality is defined by an objective function.
- *Unsupervised learning*: Given some unlabelled samples, we aim to divide them into multiple clusters or groups, of which the samples in the same group are as similar as possible but samples in different groups are as different as possible.

For a function $g(w)$ of $N$ dimensional independent variables $w \in R^{N}$, the optimisation problem is
$$argmin ~g(w)$$
a $w$ is the local minimum if it satisfies the **first-order necessary condition for optimality**
$$\huge \triangle _{w}g(w^{*}) = 0_{N\times1}$$
The point which satisfies this condition is also called **stationary point**.

![[Pasted image 20220314204537.png]]

### Gradient descent

It's a first order iterative optimisation algorithm for finding a local minimum of a differentiable cost function.
To employ the negative gradient at each step to decrease the cost function.

**The negative function consists of**
- *a direction* - determined by the gradient at the point
- *a magnitude* - sometimes called step size

Intuition: Start at any value of parameter $\theta$, then change $\theta$ in the direction that decreases the cost function and keep repeating until there is only the tiniest decrease in cost with each step.
The negative gradient of a cost function $J(\theta)$ as
$$-\triangle_{\theta}J = - \frac{dJ(\theta)}{d\theta}$$
then we need to choose a magnitude or step size parameter $\eta$ (also called **the learning rate**). Then the new **negative gradient formula** is
$$\theta(t+1) = \theta(t) - \eta \triangle_{\theta}J(\theta(t)) = \theta(t) - \eta \frac{dJ(\theta (t))}{d\theta}$$

## Logistic Regression and Maximum Likelihood

### Classification problems
#### Classification 
- def 1: determining the most likely class that an input pattern belongs to.
- def 2: modelling the posterior probabilities of class membership (dependent variable) conditioned on the input (independent) variables
#### Artificial neural networks
One output unit for each class, and for each input pattern we have
- 1 for the output unit corresponding to that class
- 0 for all the other output units
#### Binary classification
The simplest case: one output unit
#### Logistic regression
It is a regression model where the prediction (dependent variable) is categorical, e.g. binary
- **Goal**: to predict the probability that a given example belongs to the "1" class versus the probability that it belongs to the "0" class
- **Algorithm**: use the logarithm of the odds (called **logit** or **log-odds**) to model the binary prediction (dependant variable) as a linear combination of independent variables, then use the logistic function: converts log-odds to probability.

#### Odd ratio vs Probability
**Probability**: a number $p \in [0,1]$ between 0 and 1 to describe how likely an event is to occur , or how likely it is that a proposition is true.
**Odds**:  a number to describe to the probability of **a binary outcome**. The odds are the ratio of probability that an outcome presents. 
**Logit (log-odds)**: the logarithm of the odds:
$$logit(p)= log\left(\frac{p}{1-p}\right)= log(p) - log(1-p) = -log(\frac{1}{p} - 1)$$


#### Linear regression 
Linear regression (function approximation): approximating an underlying linear function from a set of noisy data.

**Problem definition** we have N observations i.e. $\{(x_{i}, y_{i})\}$ ,  $i = 1, \dots , N$ where
- $x_{i}$ - independent variables, also called regressor which  is a K dimensional vector $x_{i}\in R^{K}$, and $x_{i}=[x_{i_{1}} x_{i_{2}} x_{i_{3} \dots x_{i_{k}}}]^{T}$
- $y_{i}$: dependent variable, which is a scalar $y_{i} \in R^{1}$

**Aim**: To approximate a linear regression model to predict the dependent variable
$$\hat{y_{i}} = \theta_{0} + \sum\limits_{k=1}^{K}\theta_kx_{ik} + \epsilon_{i}~~~i=1,2, \dots , N$$
where $\theta_{0}$ is the interception and $\theta_{k}$ is the disturbance term or error variable.

#### Maximum Likelihood Estimation of Logistic Regression
Solving logistic regression: we need to estimate the $K+1$ unknown parameters $\theta$ equation 1. 

To find the maximum likelihood estimation we need to find the set of parameters for which the probability of the data is the greatest.

# Extra

## Poisson Distribution
Poisson Distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. 
$$f(x)=\frac{\lambda ^{x}}{x!}e^{- \lambda}$$

---
Resources
- [[Note-L7.pdf]]
- [[Note-L8.pdf]]
- [[Week 5-Lecture-1-Maximum Likelihood Estimation.pdf]]
- [[Week 5 Lecture-2-Maximum Likelihood Estimation and Logistic Regression.pdf]]