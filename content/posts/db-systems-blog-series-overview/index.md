+++
title = 'Database Systems - Series Overview'
date = 2024-10-22
author = 'Gabriel Stechschulte'
categories = ['database-systems']
draft = false
+++

A blog series consisting of my notes on the Carnegie Mellon University (CMU) Introduction and Advanced Database Systems Lectures by Andy Pavlo and Jignesh Patel. The primary goal of this series is to: (1) consolidate my notes, and (2) act as a reference guide for my future self. Perhaps some readers may extract some value, but I would highly recommend watching the lectures for yourself.

The series will cover:
- Database storage
- Indexes
- Join algorithms
- Query execution and processing
- Query optimization
- Query scheduling and coordination
- Concurrency control

## OLAP database management system components

The series will primarily focus on the components of OLAP database management systems (DBMS). A recent trend of the last decade is the breakout of OLAP DBMS components into standalone services and libraries for:
- System catalogs
- Intermediate representations
- Query optimizers
- File format
- Execution engines

Given these components, the general architecture of an OLAP system is the following:

![alt](db-system-architecture.png)

Where the components perform the following functions (generally speaking):
- **Front end**. Takes in the user query and parses it into an intermediate representation using a language parser.
- **Planner**. Takes in the intermediate representation from the front-end uses the binder, rewriter, and optimizer to generate a query plan.
- **Scheduler**. Takes in the query plan, organizes worker nodes, and schedules the execution by breaking the plan up into fragments.
- **Execution engine**. Takes in plan fragments and executes them.
- **I/O service**. Takes in block requests from the execution engine and returns the data after retrieving it from the object store.
- **Catalog**. Keeps track of data locations and metadata of the data for the DBMS and communicates with all components of the DBMS except the front-end.
