# Infinitesimal
In mathematics, an **infinitesimal** or **infinitesimal number** is a quantity that is closer to zero than any standard real number, but that is not zero.
# Probability Density Function - PDF
In probability theory, a **probability density function**, or **density** of a [[Probability#Continuous Random Variables| continuous random variable]], is a function whose value at any given point in the [[Probability#Sample Space|sample space]] can be interpreted as *providing a relative likelihood that the value of the random variable would be close to that sample.*

Since the _absolute likelihood_ for a continuous random variable to take on any particular value is 0 (since there is an infinite set of possible values to begin with), the value of the PDF at two different samples can be used to infer, in any particular draw of the random variable, how much more likely it is that the random variable would be close to one sample compared to the other sample.

The PDF is used to specify the probability of the [[Probability#Random variables|random variable]] falling _within a particular range of values_, as opposed to taking on any one value.

# Differentiable
A differentiable function of one real variable is a function whose [derivative](https://en.wikipedia.org/wiki/Derivative) exists at each point in its domain.
The graph of a differentiable function has a non-vertical tangent line at each interior point in its domain and does not contain any break, angle or [cusp](https://en.wikipedia.org/wiki/Cusp_(singularity) "Cusp (singularity)").
<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Polynomialdeg3.svg/600px-Polynomialdeg3.svg.png"></center>

# Median
The median is the value separating the higher half from the lower half of the data sample.
# Quartile
A quartile divides the number of data points into four parts: or _quarters_, of more-or-less equal size.
The data must be ordered from smallest to largest to compute quartiles; as such, quartiles are a form of [order statistic](https://en.wikipedia.org/wiki/Order_statistic "Order statistic"). The three main quartiles are as follows:

-   The first quartile is defined as the middle number between the smallest number and the [[Definitions#Median|median]] of the data set. It is also known as the _lower_ or _25th empirical_ quartile, as 25% of the data is below this point.
-   The second quartile is the median of a data set; thus 50% of the data lies below this point.
-   The third quartile is the middle value between the median and the highest value of the data set. It is known as the _upper_ or _75th empirical_ quartile, as 75% of the data lies below this point.

![[output-onlinepngtools.png]]

# Variance
The variance is the expectation of the squared derivation of a [[Probability#Discrete Random variables|random variable]] from its population mean or sample mean.
Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value.

![[variance.png]]

# Skewness
Skewness is a measure of asymmetry of the probability distribution of a real-valued [[Probability#Discrete Random variables|random variable]] about its mean. 


![[skewness.png]]

# Kurtosis
Kurtosis is a measure of the tailedness of the probability distribution of a real-valued  [[Probability#Discrete Random variables|random variable]]. Like [[Definitions#Skewness|skewness]], kurtosis describes the shape of a probability distribution and there are different ways of quantifying it for a theoretical distribution and corresponding ways of estimating it from a sample from a population.

![[kurtosis.png]]

# Population
Population is the universe to which we wish to generalise. In the population, we have parameters which are true (fixed) unknown values we wish to estimate (infer).

## Sample
The sample is the finite study we perform to collect data.

# Likelihood
The probability of observing your data given a particular model i.e. [[Probability#Bernoulli Distribution|the Bernoulli distribution]]. 

![[Pasted image 20220314000704.png]]

# Epoch
The epoch is what is guiding the learning procedure. 

It is guiding how many times we are making updates to the weights while the algorithm is learning. 

An epoch of 1 means we updated the parameter value as many times as the number of samples in the training data. Typically we need many epoch for the algorithm to work.
