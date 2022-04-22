#Week9 
# Generic Algorithm - GA
Genetic algorithm (evolutionary algorithm) is a **search method** that is inspired by Charles Darwin's theory of natural evolution.

**Survival of the fittest** - it reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

# Games
We can look at games as search problems. Usually games are **adversarial search problems** - the opponent has something to say about your search strategy.

We can categorise the types of games like this:
- Randomness
	- **Deterministic** - Tic-Tac-Toe, Chess, Go
	- **Stochastic** - Poker, Mahjong
- Number of players
	- One - solitare, puzzle games
	- Two - Chess, Go
	- More - Mahjong
- Competitive
	- Zero-sum - Poker, Chess
	- Non-zero-sum - Prisoner's Dilemma 
- Can you see the state
	- Perfect information - Tic-Tac-Toe, Chess
	- Imperfect information - Poker, Mahjong

## Formalisation
- States - $S$
- Actions - $A$
- Transition function - $S \times A \rightarrow S$
- Terminal test - $S \rightarrow$ (true, false)
- Players - $P=(1, \dots , N)$
- Utilities - $S \times P \rightarrow R$
	- A utility function (also called an objective function or payoff function), defines the final numeric value for a game that ends in terminal state $S$ for a player $P$.

The scope is to find a **policy** which recommends a move for each state ($S \rightarrow A$)

## Value of the State
It is the best achievable outcome (utility) from that state. 

Ex: For n queens we can consider a binary 0 - invalid, 1 - valid State. 
![[Pasted image 20220421172841.png]]

## Adversarial search - Minimax

Deterministic, zero-sum games - one player maximises the result and the other minimises the result (chess, go, tic-tac-toe).

**The minmax value** of a node is the utility of the terminal state which both players play optimally from that node.

**Minimax search:**
- A state-space search tree
- Players alternate turns
- Compute each node's minimax value (the best achievable utility against an optimal adversary)

**Implementation:**

``` C
function minimax_value (state) return its minimax value

      if state is a terminal state
            return its utility

      if state is for agent Max to take an action
            return max_value(state)

      if state is for agent Min to take an action
            return min_value(state)
```

``` C
function max_value (state) return its minimax value v

      initialise v=-∞
      for each successor of state
            v=max⁡[(v,minimax⁡_value(successor))]
      return v
```

``` C
function min_value (state) return its minimax value v

      initialise v=+∞
      for each successor of state
            v=min⁡[(v,minimax⁡_value(successor))]
      return v
```

Ex:
![[Pasted image 20220421181721.png]]
So the optimal path is the first node and the max utility we can get is 3.

## Computational Complexity of Minimax
It is basically an exhaustive DFS:
- Time: $O(b^{m})$ where $b$ is the **branching factor** and $m$ is the **maximum depth** of the tree.
- Space: $O(bm)$

![[Pasted image 20220421182556.png]]

So for some things the exact solution is infeasible, but we can optimise it. 

### Alpha-Beta Pruning
As we want the best option assuming a capable opponent, we don't have to calculate the values which can led to a worst result so we improve the efficiency by half.

![[Pasted image 20220422093643.png]]

![[Pasted image 20220422231017.png]]

**General configuration** - for agent Max
- Let $a$ be the value that Max can currently get at least.
- We are now computing the min_value at some node $n$.
- When we explore n's children , if we find that the value of n will never be better than a, then we can stop considering that branch.

**Properties of Alpha-Beta Pruning**
- The pruning has no effect on the minimax value for the root
- Good children ordering improves effectiveness of pruning
- Complexity of perfect ordering: $O(b^{m/2})$


![[Pasted image 20220422212437.png]]


# Online session

## Flow chart for genetic algorithms
When designing a genetic algorithms we should follow these steps:
- **Representation**
- **Initialisation**
- **Evaluation**
	- **Selection**
	- **Variation**
		- **Mutation**

After writing our representation we randomly generate a few models. After evaluating each of them we select the best performing ones and start generating offspring based on the selected models. We can either alternate the genes and have a chance of mutation or crossover. Repeat this process until the result is good enough and return the best solution. [[genetic-ex.pdf|This is one example.]]

## Designing a representation

There are many ways to design a representation, but the way we do must be relevant to the problem we are solving. When choosing a representation we must bear in mind what the genetic operators (crossover and mutation) might be.

![[Pasted image 20220422223747.png]]

Here we have to represent this chromosome in a different domain. The genotype represents 51 and in a 8 bit we can hold values from 0 to 255.

![[Pasted image 20220422225442.png]]
![[Pasted image 20220422225935.png]]