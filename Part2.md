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
We say a sequence of random variables $X_{1}, X_{2}, \dots, X_{N}$ are independent iff the [[Part2#Joint PDF|joint PDF]] can be factorised
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
According to the definition, *the conditional PMF* is the division of [[Part2#Joint PMF|the joint PMF]] and [[Part2#Marginal PMF and Marginal PDF|the marginal PMF]].

![[Pasted image 20220313144832.png]]

### Conditional PDF
Let $X$ and $Y$ be two [[Probability#Continuous Random Variables|continuous random variables]]. The conditional PDF of $X$ given $Y$ is
$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_{Y}(y)}$$
**The conditional PMF** can be interpreted as the [[Probability#Conditional Probability|conditional probability]] per unit. Therefore, **the conditional PDF** can take values larger than 1. This is consistent with the fact that PDF can be interpreted as the probability per unit.

## Statistical Inference

---
Resources
- [[Note-L7.pdf]]
- [[Note-L8.pdf]]