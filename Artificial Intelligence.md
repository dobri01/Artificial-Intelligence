# Probability - Artificial Intelligence 2022
## Week 1
### Signs
$\cup$ <- or, union
$\cap$ <- and, intersection
$A^c$ <- the complement of A (where A doesn't occur)
### Sample space and Events
#### Random Experiment
A random experiment is a process that produces uncertain outcomes from a well-defined set of possible outcomes.
#### Sample Space
The set of all possible outcomes of an experiment is called the Sample Space and is denoted by omega - $\Omega$.
#### Event
An event is any subset of the [[Artificial Intelligence#Sample Space|Sample Space]]. An event A is said to have occurred if the outcome of the random experiment is a member of A. They are basically sets
#### Event Space
The collection of all [[Artificial Intelligence#Event|events]] is called the Event Space, denoted as F
### Partition
A collection of sets {A<sub>1</sub>, ... , A<sub>n</sub>} is a partition to the universal set $\Omega$ if it satisfies the following conditions:
- don't overlap: {A<sub>1</sub>, ... , A<sub>n</sub>} is disjoint
- decompose: A<sub>1</sub> $\cup$ A<sub>2</sub> $\cup$ ... $\cup$ A<sub>n</sub> = $\Omega$ 
### Probability
#### ADDITIVITY of disjoint events
If $A_1,A_2...$ is a collection of **disjoint events** then:
$$P(\cup_{i=1}^\infty A_i) = \sum_{i=1}^\infty P(A_i)$$
$$ \mbox{example: }P(A\cup B \cup C) = P(A) + P(B) + P(C)$$
#### Probability Space
Consists of the [[#Sample Space|sample space]], [[#Event Space|event space]] and probability law.
#### Probability Laws
- If A $\subset$ B then P(A) <= P(B)
- P(a) = $\frac{\mbox{outcomes in }A}{\mbox{outcomes in }\Omega}$

##### Probability of intersection
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
	$$P(A \cup B \cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)$$
##### Complement Rules
$$P(A_1 \cup A_2 \cup ... \cup A_n) = 1 - P(A_1^c \cap A_2^c \cap ... A_n^c) = 1 - P(A_1^c)P(A_2^c)...P(A_n^c)$$
$$ P(A^c) = 1 - P(A)$$
#### Conditional Probability
The conditional probability is the probability of A$\cap$B out of the probability of B. Let P(B) > 0. The conditional Probability of A, given B is defined as $$ P(A|B) = \frac{P(A\cap B)}{P(B)}$$
#### Law of Total Probability
Let A<sub>1</sub>, A<sub>2</sub>, ..., A<sub>n</sub> be a partition of sample space $\Omega$. Let B be any event. Then $$P(B) = \sum_{i=1}^{n} P(A_i \cap B) = \sum_{i=1}^{n} P(A_i)P(B|A_i)$$
#### Product Rule
Suppose we have a sequence of events $A_i,...,A_n$. The probability of the intersection is equal to $$P(A_i \cap ... \cap A_n) = P(A_1)P(A_2|A_1) ... P(A_n|A_1 \cap ... \cap A_{n-1})$$
## Week 2
### Bayes’ Formula
Let A and B be two events. If P(A) != 0, then: $$P(B|A) = \frac{P(A|B)P(B)}{P(A)} = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A|B^c)P(B^c)}$$
General case:
$$P(B_j|A) = \frac{P(A \cap B_j)}{P(A)}=\frac{P(A|B_j)P(B_j)}{\sum_i P(A|B_i)P(B_i)}$$
We now provide some explanations on Bayes’ formula. In Bayes’ formula, the event A can be interpreted as **effect** and the event B can be interpreted as **cause**. Bayes’ formula tells us how to compute P(B|A), which is the probability of cause under the observation of an effect. **That is, when we observe an effect we want to find which causes this to happen**. To apply Bayes’ formula, we require the information of P(B) and P(A|B); P(A|B<sup>c</sup>).
-    P(B) is a prior probability. It is a guess of how likely the cause would occur without any observation.
-    P(A|B); P(A|B<sup>C</sup>) are conditional probability of A (effect) given B or B<sup>C</sup> (cause).
### Independence
independence means information about A does not give you any information about B (B<sup>c</sup>).
We say two events A and B are independent if (we use the notation A $\perp\!\!\!\perp$ B or A $\perp$ B). $$P(A\cap B) = P(A)P(B)$$
In this case (assuming P(B) $\ne$ 0) <- the above formula derives from the one below $$P(A|B) = \frac{P(A \cap B)}{P(B)}= \frac{P(A)P(B)}{P(B)} = P(A)$$
We have 2 ways of checking whether two events are independent: one based on the [[Artificial Intelligence#Probability of intersection| probability of intersection]] and one based on the concept of [[Artificial Intelligence#Conditional Probability|conditional probability]].
#### Pairwise Independence and Independency
If we consider more then two events, then we can have other definitions of independency. We can define the **pairwise independency** and **mutual independency**. Note that the mutual independency is stronger then pairwise independency. (we often omit the word mutual)
Consider three events A,B,C:

- We say A,B,C are pairwise independent iff $$P(A\cap B) = P(A)P(B)$$ $$P(B\cap C) = P(B)P(C)$$ $$P(A \cap C)=P(A)P(C)$$
- We say A,B,C are mutually independent (or simply said independent) if [[Artificial Intelligence#Independancy]|this]] holds and 
$$P(A \cap B \cap C) = P(A)P(B)P(C)$$
#### Application of INDEPENDENCY
If $A_1, ..., A_n$ are independent, the [[Artificial Intelligence#Product Rule|product rule]] can be simplified as follows $$P(A_1, \cap ... \cap A_n) = P(A_1)P(A_2)...P(A_n)$$
#### Independence and DISJOINTNESS
![[Pasted image 20220303160218.png|]]
- If A and F are **independent** then $P(E \cap F) = P(E)P(F)$. Let us consider the left figure. $$Left:\quad P(E) = \frac {2}{6}\quad P(F) = \frac{3}{6}\quad P(E \cap F) = \frac{1}{6}$$
This shows the independency between E and F. However, E and F are not disjoint.
- If E and F are disjoint then $P(E \cap F) = P(\emptyset) = 0$. Let us consider the right figure. $$Right:\quad P(E) = \frac {2}{6} \quad P(F) = \frac {4}{6} \quad P(E \cap F) = 0$$
Therefore, $P(E \cap F) \ne P(E)P(F)$ and **E, F are not independent but they are disjoint**.
#### Independence Extends to Complements
$$\text{A and B are independant}\iff A^c \text{ and B are independent}$$
$$\iff A \, and \, B^c \text{ are independent}\iff A^c \, and \, B^c \text{ are independent}$$

### Conditional Independency
We say A and B are conditionally independent given C iff (we write this as $A \perp\!\!\!\perp B|C$):
$$P(A\cap B |C) = P(A|C)P(B|C) \iff P(A|B,C) = P(A|C)$$
### Product rule - conditional independent
An application of conditional independency is to simplify the [[Artificial Intelligence#Product Rule|product rule]]. If $A \perp\!\!\!\perp B|C$ then
$$P(A|B,C)=P(A|C)$$
The new product rule if $A \perp\!\!\!\perp B|C$ is $$P(A_i \cap ... \cap A_n) = P(A_i)P(A_2|A_1)P(A_3|A_2)...P(A_n|A_{n-1})$$
## Week 3
### Random variables
A random variable $X$ is a function $X:\Omega \implies R$ that maps an outcome $\xi \in \Omega$ to a number $X(\xi)\in R$. We use $X(\Omega)$ or $R_X$ to denote the range of $X(\Omega) = \{X(\xi) : \xi \in \Omega\}$. The range consists of all the possible realisations of a random variable in the experiment. **We have two types of random variables depending on the range:**
- X is a discrete random variable if $X(\Omega)$ is **countable**
- X is a continuous random variable if $X(\Omega)$ is **uncountable**
### Probability Mass Function - PMF
The **Probability Mass Function** is the probability for a random variable **X** to take a value **a**
$$P_X(a) = P(X=a)$$
Note there are 2 functions here
- Function $X$: the random variable which translates words to numbers
- Function $P_X$: the mapping from the event {$X=a$} to a probability. This defines the probability for the random variable.
Note it involves two variables $X$ and $a$
- X is a **random variable** (a map). Technically if should be $X(\xi)$
- a is a **state** which is not random. The notation $X=a$ means $X$ is taking the state $a$

A PMF $P_X$ should satisfy the condition $\sum_{x \in X(\Omega)}P_X = 1$.
### Cumulative Distribution Function - CDF
The Cumulative Distribution Function $F_X(x)$ is the probability for the [[Artificial Intelligence#Random variables|random variable]] $X$ to be at most $x$.
$$F_X(x) = P(X \leq x)$$
![[Pasted image 20220311113058.png|]]
It is interesting to show the connection between [[Artificial Intelligence#Probability Density Function - PDF|PDF]] and *CDF*. One can get the other quantity if given an quantity: one direction is by integration and the other is by differentiation.
#### Remark
- If $X$ is a [[Artificial Intelligence#Continuous Random Variables|continuous random variable]] and $a\leq b$, then **integrate**
$$\int_a^b f_x(x) ~ dx = P(a \leq X \leq b) = F_X(b) - F_X(a)$$
- If $F_X$ is [[Definitions#Differentiable|differentiable]] at x, then **derivate**
$$f_X(x) = \frac{dF_X(x)}{dx} = \frac{d}{dx} \int _{-\infty}^x f_X(y) ~ dy$$
#### Proprieties
- $F_X(x)$ is a **non decreasing function** of x
- $F_X(-\infty)=0$ and $F_X(\infty) = 1$
- $P(a <X\leq b) = F_X(b) -F_X(a)$

### Special Distributions
#### Bernoulli Distribution
**Bernoulli Distribution** is related to a random variable which has only two outcomes: a success or a fail. Suppose we have a coin where the probability of heads is p and we define the random variable
$$X = \text {"The number of heads showing on one tossed coin"}$$
Then we say that $X$ is distributed according to the Bernoulli Distribution with the parameter p and write this as:
$$X \sim \text {Bernoulli (p)} $$
##### Binomial Random Variable
The Binomial Random Variable is a sum of Bernoulli random variable. We define
$$X =\text {the number of heads in an independent coin tosses with probability p of heads}$$
In general, there are $C_k^n$ sequences with k heads. Each sequence has probability $p^k(1-p)^{n-k}$ where each head happens with probability p, each tail happens with probability $1-p$. Then, the probability of having k heads out of n coin toss is:
$$P[X=k~|~n] = C_k^n~p^k(1-p)^{n-k}$$
##### Binomial Distribution
X is the number of successes in n independent trials with success probability p on each trial: $X=X_1 + X_2~...~X_n$ where $X_1 \sim \text{Bernoulli(p)}$.
$$P_X(k)=B(k;n,p) = C_k^n p^k(1-p)^{n-k}$$
<center> where $B(n,p)$ is the binomial distribution with parameter n and p . </center>

##### Geometric Distribution
Let X be the number of trials that appear until the first success. 
$X=k$ means that the first $k-1$ trials all lead to fails while the last trial is a success.
We say that $X \sim \text {Geometric(p)}$ with the range $X(\Omega) = \{1,2,3,...\}$ iff:
$$P_X(k) = (1-p)^{k-1}p ~~~~~~\text {  for k = 1,2,3,... }$$

### Continuous Random Variables
A random variable $X$ with an uncountable range $X(\Omega)$ is a continuous random variable.

**Differences**:
- we can only think about intervals
- the probability of any particular number is 0 (as the space is infinite).

A random variable having a continuous [[Artificial Intelligence#Cumulative Distribution Function - CDF|CDF]] is said to be a continuous random variable. 
Since we can't assign the probability to a specific value, we assign it according to its relative size:
$$P(\{X \in A \})= \frac{\text {"size of A"}} {\text{"size of } \Omega \text "}$$
$$P(\{X \in A \})= \int_A f_x(x)~dx$$
If you compare it with [[Artificial Intelligence#Probability Mass Function - PMF|PMF]], we note that when X is discrete, then
$$P(\{X \in A \})=\sum_{x \in A} P_X(x)$$
**Discrete** and **Continuous** variables:
- the probability of $X \in A$ is computed via *an integral* over A for *continuous random variables*
- the probability of $X \in A$ is computer via *a summation* over A for *discrete variables*

![[Pasted image 20220310235514.png|]]

*(Left)* A probability mass function ([[Artificial Intelligence#Probability Mass Function - PMF|PMF]]) tells us the relative frequency of a state when computing the probability.
*(Right)* A probability density function ([[Definitions#Probability Density Function - PDF|PDF]] or [[Artificial Intelligence#Probability Density Function - PDF|PDF]]) is the infinite [[Definitions#Infinitesimal|infinitesimal ]] version of the [[Artificial Intelligence#Probability Mass Function - PMF|PMF]]. Thus the "size" of A is the integration over the [[Definitions#Probability Density Function - PDF|PDF]].

#### Probability Density Function - PDF

Let $X$ be *continuous random variable*. The probability density function of $X$ is a function $f_X:R\implies R_+$ when integrated over an interval $[a,b]$, yields the probability of obtaining $a\leq X\leq b$
$$P(a \leq X \leq b) = \int_a^b f_x(x) ~~dx$$


### Common Continuous Random Variables
#### Uniform Random Variable
We say $X$ is a continuous uniform random variable on $[a,b]$ if the [[Artificial Intelligence#Probability Density Function - PDF|PDF]] is

$$
f_x(x) =
\begin{cases}
\frac{1}{b-a}, &if~a\leq x \leq b \\
0, &otherwise
\end{cases}
$$

We write $X \sim Uniform(a,b)$

![[Pasted image 20220311113023.png|]]
The [[Artificial Intelligence#Cumulative Distribution Function - CDF|CDF]] of a uniform random variable is

$$ F_x(x) =
\begin{cases}
0, & if ~x < a \\
\frac{x-a}{b-a}, & if ~ a \leq x \leq b \\
1, & otherwise \\		
\end{cases}
$$

#### Exponential Random Variable - MEMORYLESS property
Exponential random variable occurs if a random variable has the **memoryless of exponential random variable** property
$$P(X>x+a |X>a) = P(X>x)$$
Exponential random variables are often used to model the waiting time of some events.

We say X is an exponential random variable of parameter $\lambda$ if the [[Artificial Intelligence#Probability Density Function - PDF|PDF]] is
$$f_X(x)=
\begin{cases}
\lambda e^{-\lambda x}, &if~x \leq 0 \\
0, & otherwise
\end{cases}$$
We write $X \sim Exponential(\lambda)$
The [[Artificial Intelligence#Cumulative Distribution Function - CDF|CDF]] of an *exponential random variable* can be determined by
$$
F_{X(x)}= \int_{-\infty}^{x} f_{X}(t) dt = \int_{0}^{x} \lambda e^{-\lambda t} dt = 1 - e^{-\lambda x} ,x\geq0
$$
Therefore, the [[Artificial Intelligence#Cumulative Distribution Function - CDF|CDF]] is
$$
F_{X}(x) =
\begin{cases}
0, & if ~ x < 0 \\
1 - e^{-\lambda x} & ,otherwise
\end{cases}
$$
![[Pasted image 20220311144825.png|]]
$$\text{The PDF and CDF of } X \sim Exponential(\lambda)$$ 

**Observation:** The memoryless of exponential random variables only depends on $x$ and not on $a$.
#### Gaussian Random Variable
We say $X$ is a Gaussian random variable if the [[Artificial Intelligence#Probability Density Function - PDF|PDF]] is
$$\huge f_{X}(x)=\frac{1}{\root  \of {2 \pi \sigma}}e^{-\frac{(x-\mu)^2}{2\sigma^{2}}}$$
where $(\mu , \sigma ^{2})$ are the parameters of the distribution.  We write that
$$X \sim Gaussian(\mu,\sigma^{2}) ~ ~ ~ ~ $$
We can see that the two parameters control the shape of the Gaussian random variable. If $X \sim Gaussian(\mu,\sigma^{2})$ then
- it is symmetric around $\mu$
- $\sigma ^{2}$ determines how sharply the variable is around its centre 

 ![[Pasted image 20220311152033.png]]

#### Standard Gaussian Random Variable
We say $X$ is a **standard** Gaussian random variable if the [[Artificial Intelligence#Probability Density Function - PDF|PDF]] is
$$\huge f_{X}(x) = \frac{1}{\root \of {2 \pi}} e^{-\frac{x^{2}}{2 \sigma^{2}}}$$

