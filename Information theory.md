#Week6
# Information theory - how to quantify information

- Feature selection:
	- Information theoretic feature selection: **conditional likelihood maximisation: a unifying framework for information theoretic feature selection**
	- Clinical decision making: choose a test that provides the most information
- Unsupervised learning: mutual information criterion for clustering
- Supervised learning
	- For deep learning, information bottleneck method, a method in information theory provides a plausible theoretical foundation [Information bottleneck](https://en.wikipedia.org/wiki/Information_bottleneck_method).
	- Decision-tree learning: information theory provides useful criterion for choosing which property to split on.
	- In Bayesian learning, information theory provides a basis for deciding which is the best model given some data.
	- In logistic regression and neural networks, cross entropy is used as the loss function.

## Self Information and log-odds
Given some event $x$ and $p(x)$ is the probability of $x$ concurring, and that $p(\neg x)=1-p(x)$ is the probability of $x$ not occurring.
$$logit=log(\frac{p}{1-p})=log(p)-log(1-p)$$
we have
$$logit(x) = l(\neg x)-l(x)$$

## Entropy
It quantifies the uncertainty in a [[Probability#Discrete Random variables | random variable]] $X$. More formally, given a [[Probability#Discrete Random variables|discrete random variable]] $X$ with range $P_{X}=\{x_{1}, \dots , x_{n}\}$, and its [[Probability#Probability Mass Function - PMF|probability mass function]] as $P_{X}(x)$, **the entropy** of $X$ is formally defined as:
$$H(X)= E[I_{X}(x)]= - \sum\limits _{i}^{n}P(X=x_{i})log_{b}P(X=x_{i}) = E\left[log_{b} \frac{1}{P_{X}(x)}\right]= -E[log_{b}P_{X}(x)]$$

### Joint / Conditional entropy

**Notations**:
- The marginal PMF of $X$: $P(X)$
- The marginal probability of $X$ takes the value $x$: $p(x)$
- Joint PMF: $P(X,Y)$
- The value of joint PMF $p(X,Y)$ at $(x,y)$: $p(x,y)$
- Conditional PMF: $P(X|Y)$
- The value of conditional PMF $p(X|Y)$ at $(x,y)$: $p(x|y)$

Entropy can be extended to two or more random variables.

#### Joint entropy
**Joint entropy** is a measure of the uncertainty associated with a set of variables. For two discrete random variables $X$ and $Y$, the joint entropy is defined as:
$$\huge H(X,Y) = -E[log~p(X,Y)]=-\sum\limits _{x_{i}\in R_{X}}\sum\limits _{y_{j}\in R_{Y}} p(x_{i},y_{j}) log~p(x_{i}, y_{j})$$

The first entropy for two random variables is called joint entropy. Informally, it is a measure of the uncertainty associated with a set of variables. 

Joint entropy measures the amount of information needed on average to specify the value of more then one discrete random variables.

#### Conditional entropy

**Conditional entropy** quantifies uncertainty of the outcome of a random variable $Y$ given the outcome of another random variable $X$, which is defined as:

$$\huge H(Y|X) = -E[log~p(Y|X)]=-\sum\limits _{x_{i}\in R_{X}}\sum\limits _{y_{j}\in R_{Y}} p(x_{i},y_{j}) log~p(y_{i}| x_{j})$$

- Conditional entropy is based on the conditional PMF. Conditional entropy quantifies uncertainty of the outcome of a random variable $Y$ given the outcome of another random variable $X$
- Conditional entropy indicates that, on average, how much extra information you still need to supply to communicate $Y$ given that the other party knows $X$.
- Formally it is defined as the expectation of the conditional PMF of $Y$ given $X$.

- We can denote $H(Y|X=x)$ as the entropy of the discrete random variable $Y$ conditioned on the discrete random variable $X$ taking a certain value $x$. Then, the conditional entropy $H(Y|X)$ is the result of averaging $H(Y|X=x)$ over all possible values $x$ that $X$ may take.

##### Chain rule for conditional entropy
$$H(Y|X)=H(X,Y)-H(X)$$
$$H(X|Y) = H(X,Y) - H(Y)$$

![[Pasted image 20220407163834.png]]

### Relative entropy - Kullback-Leibler divergence (KL divergence)
It quantifies the distance between two [[Probability#Special Distributions|probability distributions]]

Let $P(x)$ and $Q(x)$ are two probability distributions of a discrete random variable $X$. That is, both $P(x)$ and $Q(x)$ sum up to 1, and $p(x)>0$ and $q(x)>0$. For any $x \in R_{X}$, the **KL divergence** from $P$ to $Q$ is defined as:
$$D_{KL}(P||Q) = \sum\limits_{X \in R_{X}}P(x)log \frac{P(x)}{Q(x)} = E[log \frac{P(x)}{Q(x)}]$$

In order for math to work: $0~log \frac{0}{Q} = 0$ and $P~log \frac{P}{0} = \infty$

It is a measure of distance so:
- $D_{KL}(P||Q)\ge 0$
- $D_{KL}(P||Q)=0$ iff $P(x)=Q(x)$

It is not a true distance:
- $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

This is used in machine learning such as supervised learning and reinforcement learning.

#### Other distance measures

##### Cross Entropy

For two discrete distributions $P$ and $Q$, cross entropy is defined as
$$H(P,Q)=- \sum\limits_{x \in R_{X}}P(x)log~Q(x) = H(P)+D_{KL}(P||Q)$$
##### Jensen-Shannon divergence (JSD)
A symmetrized and smoothed  version of the [[Information theory#Relative entropy - Kullback-Leibler divergence KL divergence| Kullback-Leibler divergence]] $D(P||Q)$. 
$$JSD(P||Q)=\frac{1}{2} D_{KL}(P||M)+ \frac{1}{2} D_{KL}(Q||M)$$ where $M=\frac{1}{2} (P+Q)$

##### Wassertein Distance - Earth Mover's distance - [[1904.08994.pdf | Generative Adversarial Networks]]
The minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution.

## Mutual information
Measures the information that $X$ and $Y$ share. It measures how much knowing one of these variables reduces uncertainty about the other.

It is defined as follows for two discrete random variables $X$ and $Y$:
$$I(X;Y) = \sum\limits_{x \in R_{X}}\sum\limits_{y \in R_{Y}} p(x,y)~log~\frac{p(x,y)}{p(x)p(y)}$$

Used in feature selection 

### MI and KL divergence
The mutual information $I(X;Y)$ is also defined as the KL divergence between the joint distribution and the product of marginal distributions.
$$I(X;Y)= D_{KL}(P(X,Y)||P(X)P(Y))$$

The mutual information essentially measures the distance (error) of using $P(X)P(Y)$ to model the joint probability $P(X,Y)$. When $X$ and $Y$ are independent of each other, i.e. $p(x,y)=p(x)p(y)$ we have
$$I(X;Y)=D_{KL}(P(X,Y)||P(X)P(Y))==0$$
If the two PMFs are truly independent, then the KL divergence is 0 (the two random variables don't share any information).

### MI and Entropy
We can also express mutual information in terms of joint and conditional entropies.
$$I(X;Y)=H(X)- H(X|Y)$$
$$I(X;Y)=H(Y)-H(Y|X)$$
$$I(X;Y)=H(X)+H(Y)-H(Y,X)$$
$$I(X;Y)=H(X,Y)-H(Y|X)-H(X|Y)$$

![[Pasted image 20220408181502.png]]

### Properties of MI
- Non-negativity: $I(X;Y) \ge 0$
- Symmetric
- Measure of dependence between $X$ and $Y$
	- $I(X;Y)=0$ iff $X \bot Y$
	- $I(X;Y)$ not only increases with the dependence of $Y$ and $Y$but also with $H(X)$ and $H(Y)$
- $I(X;Y)=H(X)-H(X|X)=0$

#Week7

# Decision tree learning

Predictive model: a tree structure consists of:
- **Root or internal node**: a independent variable (a feature or an attribute)
- **Leaf**: an outcome of the dependent variable.
- **Branch**: a rule (or decision)

Types:
- **Classification trees**: dependent variables are categorical or qualitative (male/female)
- **Regression trees**: dependent variables are continuous or quantitative (temperature)

Popular algorithms:
-    [ID3 (Iterative Dichotomiser 3)](https://en.wikipedia.org/wiki/ID3_algorithm)
-    [C4.5 (successor of ID3)](https://en.wikipedia.org/wiki/C4.5_algorithm)
-    [CART (Classification And Regression Tree)](https://www.nature.com/articles/nmeth.4370)

## Entropy for decision trees
**Q:** How to construct a decision tree? How to learn the structure from data?

**A:** Given a data set, the algorithm groups and labels samples that are similar between them    and look for the best rules that split the samples  
that are dissimilar between them until they reach certain degree of  
similarity.

**Q:**    How to look for the best rules that split the samples?

**A:** two main methods:
- **Gini index** (Gini impurity) - used in CART
- **Information gain** - used in ID3 and C4.5

The **Gini index** given a training dataset of $J$ classes
$$I_{G}(p)=1- \sum\limits_{i=1}^{J}p_{i}^{2}$$ where $p_{i}$ is the fraction of items labelled with class $i$ in the dataset. 

## Information gain
**The information gain** is the information we can gain after spiting the samples based on a independent variable (internal node). Formally, **information gain** is defined as the change in information entropy $H$ from a prior state to a state that takes some information as given:
$$IG(Y,X)=H(Y)=H(Y|X)$$ where $Y$ is a random variable that represents the dependent variable and $X$ is one of the independent variables, and $H(Y|X)$ is the conditional entropy of $Y$ given $X$.