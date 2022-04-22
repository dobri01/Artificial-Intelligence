#Week8

# Search problems
Search is not about prediction but about choice and decision-making. 
- path-finding - Google Maps
- taking next move in games - AlphaGo
- task assigning - Uber

Search is the process of navigating from a start state to a goal state by transitioning through intermediate states.

A search problem consists of
- **state space** - all possible states
- **start state** - where the agent begins the search
- **goal test** - whether the goal state is achieved or not
- **successor function** - given a certain state, what actions are available and for those actions what are the next stages the agent will land into.

# State space graph and Search Tree
State space graph
- nodes represent states
- arcs represent successors
- goal test is a goal node or a set of goal nodes

![[Pasted image 20220415163308.png]]

Search tree
- the start state is the root node
- children correspond to successors
- nodes show states, but correspond to plans that achieve those states

![[Pasted image 20220415163513.png]]

# Generic tree search 
``` d
function Tree-Search (problem, strategy) return a solution, or failure  
initialise the search tree via setting Frontier to be the start state of problem  
loop do  
if there are no nodes in Frontier for expansion then return failure  
choose a node in Frontier for expansion according to strategy and remove it  
from Frontier  
if the chosen node is a goal state then return the corresponding solution  
else expand the node based on problem and add the resulting nodes to Frontier  
end
```
 Important things
 - Frontier
 - Expansion
 - Expansion strategy 

## Depth-First Search - DFS

![[Pasted image 20220415164230.png]]

## Breadth-First Search - BFS
![[Pasted image 20220415164307.png]]

# Identification problem and CSP

Search can also be used for planning and identification.
- planning - sequence of actions (we care about the path)
- identification - an assignment (we only care about the goal)

![[Pasted image 20220415195701.png]]

# Search problems
- Constant Satisfaction Problems (CSP) - identification problems have constraints to be satisfied, there is no preference in CSPs
- Constraint refer to hard constraints which a legal solution cannot violate
- Preferences sometimes are referred to as soft constraints (or objectives) where we need to optimise (minimise cost)

# Standard Search Problems vs CSPs
Standard Search Problems
- State is a "black box" - arbitrary data structure
- Goal test can be any function over states

Constant Satisfaction Problems
- State is defined by variable $X_{i}$ with values from a domain $D$ ($D$ might depend on $i$)
- Goal test is a set of constraints specifying allowable combinations of values for subsets of variables
- An example of a formal representation language
- This allows useful general-purpose algorithms with more power then standard search algorithms

# CSP
A constant satisfaction problem consists of
- a set of variables
- a domain for each value
- a set of constraints

In a CSP, an assignment is **complete** if every variable has a value, otherwise it is **partial**.

**Solutions** are complete assignments satisfying all the constraints.

![[Pasted image 20220420140031.png]]
![[Pasted image 20220420001940.png]]

# Constraint Graphs

- Constraint graphs are used to represent relations among constraints in CSPs, where nodes correspond to the variables and arcs reflect the constraints.
- They are different from state space graphs since variables in CSPs correspond to multiple states.
- Binary CSP: Each constraint involves two variables.
- General purpose: CSP algorithms use the graph structure to speed up the search

**When a constraint relates more then two variables** 

- Use a square to represent a constraint (circle for variable), the square connects all the variables involved into that constraint 

Constraint graph for the problem above:

![[Pasted image 20220420002358.png]]

# Variety of CSPs
**Variables**
- Finite domains (discrete)
- Infinite domains (discrete or continuous)

**Constraints**
- Unary (single variable having reducing domains), binary and high-order constraints

**CSPs are difficult search problems**
- If a CSP has $n$ variables, the size of each domain is $d$, then there are $O(d^{n})$ complete assignments

**CSP examples**
- Assignment problems - who teaches each class
- Timetabling problems - which class is offered when and where 
- Hardware configuration
- Transportation scheduling
- Factory scheduling
- Circuit layout
- Fault diagnosis

Many CSP problems can also consider the preferences, in which case they turn into **constraint optimisation problems**.
# Solving CSPs
## Generate and test
The exhaustive generate-and-test algorithm is generating all the complete assignments, then testing them in turn, and returning the first one that satisfies all the constraints.

 It needs to store all $d^{n}$ complete assignments, where $d$ is the domain size and $n$ is the number of variables.

## Standard search formulation 
In CSPs, states are defined by the values assigned so far (partial assignments)
- **Initial state**: the empty assignment { }.
- **Successor function**: assign a value to an unassigned variable.
- **Goal test**: if the current assignment is complete and satisfies all the constraints.


## BFS
Not a good idea as it needs to transverse all the nodes to get to a solution 

## DFS
Would work in some applications but we need to consider the constraints as we go (so we don't waste time on a branch that is not viable)

## Backtracking Search
Backtracking is a DFS method with two bonus properties:
- **check constraints as you go**
	- consider only values that don't conflict previous assignments
- **consider one variable at a layer**
	- variable assignments are commutative, so fix ordering

### Improving Backtracking
We can use 
- **Filtering** - can we detect inevitable failure early?
- **Ordering** - which variable should we assigned next 

#### Filtering
Keep track of domains for unassigned variables and cross off bad options.

##### Forward Checking
Cross off values that violate a constraint when added to the existing assignment.

![[Pasted image 20220417161112.png]]
##### Ordering
**Consider the minimum remaining values** - choose the variable with the fewest legal values left in its domain.

![[Pasted image 20220417161308.png]]


# Tree search vs Local search
**Tree search methods** - systematically search the space of assignments
- Start with an empty assignment 
- Assign a value to an unassigned variable and deal with constraints on the way until a solution if found

If the space is too big/infinite, systematic search may fail to consider enough of the space to give any meaningful results.

**Local search methods** - not systematically search the space but designed to quickly find solutions
- Start with an arbitrary complete assignment, so constraints can be violated
- Try to improve the assignment iteratively

# Local Search for CSPs

A typical local search algorithm (hill climbing for CSP):
- Randomly generate a complete assignment 
- While the stop criteria is not met:
	- Variable selection - randomly select a constraint-violated variable
	- Value selection (min-conflict heuristic) - choose a value that violates the fewest constraints

![[Pasted image 20220418200359.png]]

Local search might get stuck in a loop if the search strategy doesn't account for it.


# Optimisation 
Optimisation problems - search problems with preferences (objective functions)

They consist of:
- **variables**
- **domains**
- **objective functions**

Objectives:
- **Single-objective optimisation problems** - ex: minimising the cost of travelling (TSP)
- **Multi-objective optimisation problems** - ex: TSP with minimum time

Constraints:
- **Unconstrained optimisation problems**
- **Constraint optimisation problems**


# Online session questions 
Question 1

![[Pasted image 20220420125124.png]]
![[Pasted image 20220420130406.png]]
![[Pasted image 20220420130632.png]]

Question 2

![[Pasted image 20220420131543.png]]
![[Pasted image 20220420132415.png]]
![[Pasted image 20220420132735.png]]
- We can always design different rules with different search strategies. 
- We can also initiate multiple assignments and deal with them in a parallel way, like in population based search

# Optimising CSP problems

Here is an optimised way of solving [[Search Trees CSP Optimisation#CSP|these two problems]].

We may be able to formulate CSP differently:
-   Encode some constraints into domains directly
- Consider some constraints when formulating the problem
## Cryptarithmetic
![[Pasted image 20220420134246.png]]

- We reduced the search space of the problem as $X1$ $X2$ $X3$ are used for addition and don't need the full domain.
- We can also put $T$ and $F$ different to 0 for constraints.
## N Queens
![[Pasted image 20220420192608.png]]

- If we encode the queens like this then we can simplify the constraints and have less variables (reduced dimension)
- We don't need the last condition from last time (we included it into the notation)

## Minesweeper
![[Pasted image 20220420194540.png]]

- Based on forward checking and ordering we can optimise in what order the variables are visited 
- This can be optimised way more