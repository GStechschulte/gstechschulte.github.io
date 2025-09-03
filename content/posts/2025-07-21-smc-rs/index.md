+++
title = '[WIP] Fast Sequential Monte Carlo Sampler in Rust'
date = 2025-06-11
author = 'Gabriel Stechschulte'
categories = ['probabilistic-programming', 'rust']
ShowToc = true
TocOpen = false  # Optional - keeps TOC open by default
draft = true
math = true
+++

In this post, it is discussed how one can use Rust's type system to implement a highly-performant Sequential Monte Carlo (SMC) algorithm. First, an overview of the SMC algorithm is given. Given this overview, I then discuss the "hot parts" of the algorithm, i.e., steps in the algorithm that hinder performance, followed by how one can use Rust's type system to circumevent these bottlenecks.

## Sequential Monte Carlo

The core algrithm of SMC is simple:

1. Update
2. Weight
3. Resample

## The Rusty bits

It was shown that the resampling step is very important as this leads to a population of particles that is more representative of the posterior distribution. The most common resampling method is _systematic resampling_. A subtle part of this method is that we are resampling with replacement. Thus, if we have five particles $[P_1, P_2, P_3, P_4, P_5]$ where the subscript indicates Particle 1, Particle 2, etc., and **not** the index. After resampling, we may have selected the following ancestors $[P_2, P_2, P_3, P_4, P_4]$. Particle 2 and 4 were sampled twice where Particle 3 was sampled once, and Particles 1 and 5 were not selected for further updating. As sampling is performed with replacement, we need to copy Particle 2 and 4 two times because in the next iteration, these particles will be updated independent of each other.

Depending on what these particles represent, this copy may be expensive due to the data type being copied. For example, if the core data structure representing a particle is a `struct`, then the entire structure would need copied.

### Copy on write

When resampling, we could defer copying the particle until we know that it will be mutated. We can implement this memory management strategy using the Clone-on-Write (COW) technique. Specifically, we can use a smart pointer such as `Rc` or `Arc` to enable shared ownership of particles.

to implement the COW technique.

Rust, COW can be implemented when using a smart pointer such as `Rc` or `Arc`.

to allow shared ownership of data while minimizing unncessary copying.


If the count is 1, it returns a mutable reference to the inner data without allocating.
If the count is > 1 (meaning it's shared), it will clone the inner data, create a new Rc with the clone, and return a mutable reference to the new data. This is called "clone-on-write" and is exactly what you need after a resampling step.

## Summary
