+++
title = 'Stateful Joins in SQL'
date = 2024-08-22
author = 'Gabriel Stechschulte'
categories = ['data-engineering']
ShowToc = true
TocOpen = false  # Optional - keeps TOC open by default
draft = false
math = true
+++

## Introduction

In some scenarios, one needs to enrich an event stream with data from another source that holds "state". This state provides additional context to the event stream.

For example, in manufacturing, a machine may use a set of machine process parameters (pressure, speed, force, etc.) when producing an item. The process parameters represent the "state" of the machine at production time $t$. However, the software services that publishes messages on what is being produced and the machine process parameters currently used are separate. Furthermore, to avoid the duplication of data, the service that publishes process parameters only publishes a message when there is a change in state, e.g when an operator changes one of process parameters.

## Data simulation

Lets simulate some data with TimescaleDB.

```sql
CREATE TABLE production (
    time timestamptz NOT NULL,
    product_id INT NOT NULL
);

INSERT INTO production
SELECT *,
   1 as product_id
FROM generate_series('2024-01-01 05:00:00', '2024-01-01 05:05:00', INTERVAL '1m') AS time
UNION ALL
SELECT *,
    2 as product_id
FROM generate_series('2024-01-01 05:10:00', '2024-01-01 05:13:00', INTERVAL '1m') AS time

SELECT * FROM production;
```

| time                  |   product_id |
|:----------------------|-------------:|
| 2024-01-01 05:00:00+00 |            1 |
| 2024-01-01 05:01:00+00 |            1 |
| 2024-01-01 05:02:00+00 |            1 |
| 2024-01-01 05:03:00+00 |            1 |
| 2024-01-01 05:04:00+00 |            1 |
| 2024-01-01 05:05:00+00 |            1 |
| 2024-01-01 05:10:00+00 |            2 |
| 2024-01-01 05:11:00+00 |            2 |
| 2024-01-01 05:12:00+00 |            2 |
| 2024-01-01 05:13:00+00 |            2 |

```sql
CREATE TABLE machine (
    time timestamptz NOT NULL,
    speed NUMERIC NOT NULL
);

INSERT INTO machine (time, speed)
VALUES ('2024-01-01 02:00:00'::timestamptz, 40.0),
       ('2024-01-01 05:07:00'::timestamptz, 60.0);

SELECT * FROM machine;
```

| time                  |   speed |
|:----------------------|--------:|
| 2024-01-01 02:00:00+00 |     40.0 |
| 2024-01-01 05:07:00+00 |     60.0 |


## Postgres stateful join

We would like to enrich the `production` data with the process parameters from `machine`. Thus, we need to join the **most recent** process parameter with a production event where a production event most occur greater than or equal to the change in machine state.

This enrichment can be achieved with a _stateful join_ using PostgreSQL's `LATERAL JOIN` expression. The `LATERAL` keyword allows a subquery or derived table to reference columns from tables listed before it in the `FROM` clause. A `LATERAL` join is like a for loop: for each row returned by the tables listed before `LATERAL` in the `FROM` clause, PostgreSQL will evaluate the `LATERAL` subquery using the current row's values. The resulting rows from the `LATERAL` subquery are joined to the current row, typically using a `JOIN` condition of `ON TRUE` since the real join conditions are inside the `LATERAL` subquery. This process is then repeated for each row or set of rows from the tables preceding `LATERAL`.

```sql
SELECT *
FROM production prod
LEFT JOIN LATERAL (
    SELECT time as change_time,
           speed
    FROM machine
    WHERE time <= prod.time
    ORDER BY time DESC
    LIMIT 1
    ) ON TRUE;
```

| time | product_id | change_time | speed |
|---|---|---|---|
| 2024-01-01 05:00:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:01:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:02:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:03:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:04:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:05:00.000000 +00:00 | 1 | 2024-01-01 02:00:00.000000 +00:00 | 40 |
| 2024-01-01 05:10:00.000000 +00:00 | 2 | 2024-01-01 05:07:00.000000 +00:00 | 60 |
| 2024-01-01 05:11:00.000000 +00:00 | 2 | 2024-01-01 05:07:00.000000 +00:00 | 60 |
| 2024-01-01 05:12:00.000000 +00:00 | 2 | 2024-01-01 05:07:00.000000 +00:00 | 60 |
| 2024-01-01 05:13:00.000000 +00:00 | 2 | 2024-01-01 05:07:00.000000 +00:00 | 60 |

In our hypothetical manufacturing example, the machine process parameters changed when product `2` began producing. Before this enrichment process, it wouldn't have been known why the time to produce product `2` was faster. However, the `LATERAL JOIN` allows us to see that the speed increased from 40 to 60.
