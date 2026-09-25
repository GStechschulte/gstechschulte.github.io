+++
title = "Technology Shifts the Bottleneck"
date = 2026-09-25
author = "Gabriel Stechschulte"
tags = ["economics"]
ShowToc = true
TocOpen = false
draft = false
+++

Imagine in your line of work, a piece of technology is introduced such that it makes it vastly more productive to produce one additional unit of output. Maybe you are a mechanical engineer specialized in CAD, or a software engineer writing code. Let's call this piece of technology the __generator__. In the presence of such a generator, what happens to the demand for people like you? What about wages? What should I focus on now? To begin thinking about such types of questions, let's first provide two perspectives: (1) task level, and (2) market level.

## Task level

At the task level, this is your job. It is composed of multiple tasks where technology is an affordance on those tasks. Very rarely does a technology encompass an entire occupation. Let's consider a software engineer at Uber. Their output depends on the amount of code written, which we will call $i$, and all other complementary work, which we will call $j$.

As the price (cost) of code approaches zero, and assuming $j$ is a complement to code (more code makes $j$ more valuable), the marginal value of $j$ increases. Thus, as code generation becomes more productive, the bottleneck shifts to the complementary work. The marginal value now lies in focusing on $j$. It is important to consider what the complementary work is. In the occupation of software engineering this can be: system design, verification of correctness, integration, or collaboration with stakeholders.

Okay, so a job is composed of tasks. If one task becomes much more productive, the bottleneck (and thus our attention) shifts from one task to another more valuable task. One thing I would like to point out is that many of the complementary tasks, such as system design and verification of correctness, require deep technical skills accumulated through years of experience. As individuals and organizations, our goal should still be to develop and understand complex systems. This still requires the acquisition of hard skills and knowledge as these are still[^1] required by complementary tasks.

## Market level

Suppose the generator augments labor, enabling us to build more software, or software we otherwise would not have[^2]. Whether employment increases depends on the elasticity of demand for _that_ software. For example, if a 10% decrease in the effective cost of building software raises the quantity demanded by more than 10%, demand is elastic, and total labor increases. This counterintuitive result is known as Jevons paradox: greater efficiency lowers the effective cost of the resource, thereby increasing demand even more. If demand is inelastic, however, the same output is produced with fewer inputs (engineers). Given this market level view, it is also interesting to think about it from the view of demand for software at _your_ company. Are there products, internal tooling, or features that are **not** built because the generation of that code is the constraint?

You also have the supply side dynamics, which mainly act on wages. Continuing with our software engineering example, the supply-side effect depends on which tasks the generator automates. If it automates the routine tasks $i$, the remaining work is concentrated in $j$, which is acquired through experience. Fewer people qualify, and those who do can expect higher wages. If instead the generator begins to automate parts of $j$ itself, the job becomes more accessible, the pool of qualified workers grows, and wages fall. Today's tools mostly do the former, but we should not ignore that people are working on tools that do the latter. An interesting dynamic to consider here (and something I pointed out above) is that if the traditional route to acquiring expert knowledge lies in the experience gained through the junior level roles, and companies stop hiring junior level engineers, then we are underinvesting in future supply.

## Summary

How a technology impacts you depends on the tasks performed within your job. The introduction of new technology changes the bottlenecks. As one input becomes abundant, another one becomes more scarce, whereby, performing task $j$ over $i$ becomes more valuable. Very rarely does a single technology encompass an entire job. However, as new technology continues to be developed, you should continually gain new skills and deepen your expertise. Time for bed, there's $j$ to do tomorrow[^3].


[^1]: I say "still" because I know there are people working on the automation of research, system design, formal verification, etc.
[^2]: It is important to note that just because software was not built does not mean that code generation was the bottleneck. Product decisions, customer adoption, or regulatory approval can all be reasons for why something is not built.
[^3]: And new skills to learn and hone.
