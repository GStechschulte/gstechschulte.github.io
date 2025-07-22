+++
title = '[WIP] Gymnastics in Constrained Optimization'
date = 2025-06-11
author = 'Gabriel Stechschulte'
categories = ['optimization']
ShowToc = true
TocOpen = false  # Optional - keeps TOC open by default
draft = true
math = true
+++

## Constrained optimization

In unconstrained optimization problems, each design variable can take on any real number. However, in most applied and industrial settings, we need to perform constrained optimization which forces design points to satisfy certain conditions. A theme in the mathematical optimization field is to transform the constrained problem into an unconstrained one. This transformation process often requires some gymnastics. This blog reviews some of the popular approaches for transforming a constrained problem into problems without constraints and attempts to connect the underlying concepts these approaches utilize. In general there are two approaches to addressing constraints:

1. **Primal approach**. One attempts to modify an optimization methodß so as to stay within feasible regions of the space. Many optimization methods can be modified to stay within feasible regions of the space, e.g., projected gradient descent.
2. **Dual approach**. Uses Lagrangian relaxation in order to create a new *dual* problem in which primal constraints are converted into dual variables. In many cases, the structure of the dual problem is simpler to solve.

### Transformations to remove constraints

Transform the problem so that constraints are completely removed. This is often performed through variable transformations. For example, suppose the design variables are the acceleration and deceleration of a manufacturing machine. The values these variables can take on are bound to the operational limits of the hardware. Bound constraints $a \leq x \leq b$ can be removed by passing $x$ through a transform. For example

$$x = t_{a,b}(\hat{x}) = \frac{b + a}{2} + \frac{b - a}{2} \frac{(2\hat{x})}{1 + \hat{x}^2}$$

### Lagrange multipliers

The solution to a constrained optimization problem is restricted to a certain region or curve. Often, this solution occurs at this boundary. This can be a problem because if the maximum or minimum occrs at the boundary, then the gradient is does not equal zero.



### Augmented Lagrange method

### Interior-point method

#### Primal-dual method

### Duality

**Duality** theory in optimization establishes that every optimization problem (the **primal problem**) has a corresponding **dual problem** where "if the primal is a minimization problem then the dual is a maximization problem (and vice versa). This provides a key insight that any feasible solution to the primal (minimization) problem is at least as large as any feasible solution to the dual (maximization) problem, providing natural bounds.

The Lagrangian creates the bridge between primal and dual formulations, where the dual variables $\lambda$ represent the incremental change in the optimal solution value per unit increase in the RHS of the constraint.

The Lagrange provides an alternative framing of the objective. Where do the inequality constraints go? Why does the `max` enforce the inequality constraints?

Primal-dual methods exploit duality by simultaneously update the primal and dual.

## The connection

- Some methods exploit the concept of _duality_ to solve the optimization problem, whereas other approaches work directly.
- The introduction of the dual variable, whenever you have "bad stuff" to deal with, is pervasive in optimization and control.
- Other methods "work" primarily in the primal space.


## Other approaches

There are others methods to solving...

- Active set methods - Iteratively determine which constraints are binding and solve equality-constrained subproblems
- Sequential quadratic programming (SQP) - Solve a sequence of quadratic approximations to the original problem
- Trust region methods - Constrain the step size rather than working directly with original constraints
- Feasible direction methods - Move along directions that maintain feasibility while improving the objective
- Cutting plane methods - Add linear constraints that cut off infeasible regions
- Projection methods - Project iterates back onto the feasible set

## Conclusion
