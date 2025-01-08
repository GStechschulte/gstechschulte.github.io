+++
title = '[WIP] Database Systems - Query Execution and Processing'
date = 2025-01-01
author = 'Gabriel Stechschulte'
categories = ['database-systems']
draft = false
math = true
+++

## Operator execution

In OLAP systems, sequential scans are the primary method for query execution. The goal is two-fold: (1) minimize the amount of data fetched from the disk or a remote object store, and (2) maximize the use of hardware resources for efficient query execution.

Andy’s (unscientific) top three execution optimization techniques:
- **Data parallelization (vectorization)**. Breaking down a query into smaller tasks and running them in parallel on different cores, threads, or nodes.
- **Task parallelization (multi-threading)**. Breaking down a query into smaller independent tasks and executing them concurrently. This allows the DBMS to take full advantage of hardware capabilities and or multiple machines to improve query execution time.
- **Code specialization (pre-compiled / JIT)**. Code generation for specific queries, e.g. JIT or pre-compiled parameters.

which fall into three primary ways for speeding up queries:

1. **Reduce instruction count**. Use fewer instructions to do the same amount of work.
1. **Reduce cycles per instruction**. Execute more CPU instructions in fewer cycles, i.e. data that we need is in L1/L2/L3 cache to maximize data locaclity.
1. **Parallelize execution**. Use multiple threads to compute each query in parallel.

## Query execution

The DBMS converts a SQL statement into a *query plan*. This plan is a DAG of operators where each _operator instance_ is an invocation of an operator on a unique segment of data. A task is a sequence of one or more operator instances. A _task set_ is the collection of executable tasks for a logical pipeline.

```SQL
SELECT A.id, B.value
FROM A
JOIN B
USING (id)
WHERE A.value < 99
	AND B.value > 100;
```

TODO: Add query plan diagram here..

The distinction between operators and operator instances is made because we can have multiple operators run in parallel, e.g. if table `A` is a billion rows, we can divide up the scan operator into 10 instances where each instance scans different files or row-groups in an object store.

## Processing models

A DBMS _processing model_ defines how the system executes a query plan and moves data from one operator to the next. It specifies how things like the direction in which the query plan is evaluated and what kind of data is passed between the operators along the way. There are different processing models with various tradeoffs for different workloads, e.g OLTP and OLAP. The two different plan processing directions are:

1. **Top-to-Bottom**. Starts with the root node and "pulls data up" from the children. This approach always passes tuples with function calls.
2. **Bottom-to-Top**. Starts with leaf nodes and "pushes data up" to its parents.

The three main processing models to consider are **Iterator**, **Materialization**, and **Vectorization** where each model is comprised of two types of execution paths:

1. **Control flow**. How the DBMS invokes an operator.
2. **Data flow**. How an operator sends its results.

The output of an operator can be either whole tuples as in the N-ary storage model (row-oriented storage) or subsets of columns as in the decomposition storage model (column-oriented storage).

### Iterator model

Also known as the Volcano or Pipeline model, is a model where each query plan operator implements a `Next` function. On each invocation, the operator returns either a single tuple or an end of file (EOF) marker if there are no more tuples. The operator implements a loop that calls next on its children to retrieve their tuples and then process them. Each tuple is then processed up the plan as far as possible before
the next tuple is retrieved.

Query plan operators in an iterator model are highly composible because each operator can be implemented indepedent from its parent or child so long as it implements the `Next` function.

The iterator model also allows for *pipelining* where the DBMS can process a tuple through as many operators as possible before having to retrieve the next tuple. The series of tasks performed for a given tuple in the query plan is called a *pipeline*. However, some operators may be blocked until their children emit all of their tuples, e.g. with joins, subqueries, and order bys.

**TODO: insert diagram**

The downside with the iterator model is that we are basically calling `Next` for every single tuple. If there are a billion tuples, then there will be one billion `Next` calls.

### Materialization model

The materialization model is a specialization of the iterator model where instead of having a `Next` function that returns a single output, each operator processes all of its input and then emits its output all at once. The operator "materializes" its output as a single result. To avoid scanning too much input, the DBMS can push down hints, e.g. limits. The output can be either a whole tuple as in row-oriented storage or a subset of columns as in columnar storage.

**TODO: insert diagram**

Every query plan operator implements the `Output` function. The operator proceesses all of the tuples from its children at once. The return result of this function is all of the tuples that operator will ever emit.

This approach is better for OLTP workloads because queries typically only access a small number of tuples at a time. Thus, there are fewer function calls to retrieve tuples. The materialization model is not suited for OLAP queries with large intermediate results because the DBMS may have to spill those results to disk between operators.

### Vectorization model

The vectorization model is a hybrid approach of the iterator and materialization model. Like the iterator model where each operator implements a `Next` function, but each operator emits a batch of tuples instead of a single tuple. The operators internal loop processes multiples tuples at a time. The size of the batch can very based on hardware or query properties. Each batch will contain one or more colums with each having their own null bitmaps.

**TODO: insert diagram**

The vectorized model is considered ideal for OLAP queries because it greatly reduces the number of invocations per operator, removes tuple-navigation overhead, and allows operators to use vectorized (SIMD) instructions to process batches of tuples.

## Plan processing direction

In the previous sections, the DBMS starts executing a query by invoking a `Next` at the root of the query plan and pulls data up from leaf operators. This *pull based* is how most DBMSs implement their execution engine. However, there is also *push based*.

### Top-to-bottom (pull)

Starts with the root and pulls data up from its children. Tuples are always passed between operators using function calls (unless it's a pipeline breaker).

**TODO: insert diagram**

### Bottom-to-top (push)

Start with leaf nodes and push data to the parents. With push-based, you can fuse operators together within a for loop to minimize intermediate result staging.

**TODO: insert diagram**

## Filter representation

With the iterator model, if a tuple does not satisfy a filter, then the DBMS just invokes `Next` again on the child operator to get another tuple. In the vectorized model, however, a vector/batch may contain some tuples that do not satisfy filters. Therefore, we need some logical representation to identify tuples that are valid and need to be processed to include in the materialized results. There are two primary approaches to do this: **selection vectors**, and **bitmaps**.

### Selection filters

Selection filters are used to store the indices or identifiers of tuples that are valid and should be considered for further processing. These filters are typically a dense sorted list of tuple identifiers that indicate which tuples in a batch are valid.

**TODO: insert diagram**

### Bitmaps

A bitmap is created to indicate whether a tuple is valid and can be used as an input mask for further processing. The bitmap is positionally-asligned that indicates whether a tuple is valid at an offset. Some SIMD instructions natively use these bitmaps as input masks.

**TODO: insert diagram**
