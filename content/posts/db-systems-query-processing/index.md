+++
title = 'Database Systems - Query Processing'
date = 2024-10-23
author = 'Gabriel Stechschulte'
categories = ['database-systems']
draft = true
math = true
+++

## Query plan

The DBMS converts a SQL statement into a query plan. Operators in the query plan
are arranged in a tree. Data flows from the leaves of this tree towards the root node.
The output of the root node in the tree is the result of the query. The same query
plan can be executed in multiple ways.

```SQL
SELECT A.id, B.value
FROM A
JOIN B
USING (id)
WHERE A.value < 99
	AND B.value > 100;
```

TODO: Add query plan diagram here..

## Query execution

A query plan is a DAG of operators where each _operator instance_ is an invocation of an operator
on a unique segment of data. A task is a sequence of one or more operator instances. A _task set_
is the collection of executable tasks for a logical pipeline.


## Processing models

A DBMS _processing model_ defines how the system executes a query plan and moves data from
one operator to the next. It specifies how things like the direction in which the query plan is evaluated and what kind of data
is passed between the operators along the way. There are different processing models with
various tradeoffs for different workloads, e.g OLTP and OLAP. The three main execution models to consider are

- Iterator
- Materialization
- Vectorization

Each processing model is comprised of two types of execution paths:

1. **Control flow**. How the DBMS invokes an operator.
2. **Data flow**. How an operator sends its results.

The output of an operator can be either whole tuples as in the N-ary storage model (row-oriented
storage) or subsets of columns as in the decomposition storage model (column-oriented storage).
