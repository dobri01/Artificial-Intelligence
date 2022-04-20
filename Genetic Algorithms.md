#Week9 

# Adversarial search - Minimax

Deterministic, zero-sum games - one player maximises the result and the other minimises the result (chess, go, tic-tac-toe)

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