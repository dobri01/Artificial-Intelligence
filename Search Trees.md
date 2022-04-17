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

![[Pasted image 20220415202411.png]]

# Constraint Graphs

- Constraint graphs are used to represent relations among constraints in CSPs, where nodes correspond to the variables and arcs reflect the constraints.
- They are different from state space graphs since variables in CSPs correspond to multiple states.
- Binary CSP: Each constraint involves two variables.
- General purpose: CSP algorithms use the graph structure to speed up the search

**When a constraint relates more then two variables** 

- Use a square to represent a constraint (circle for variable), the square connects all the variables involved into that constraint 

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