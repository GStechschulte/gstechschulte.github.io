---
aliases:
- /2021/08/10/No-Code-Dependency-and-Building-Technology
date: '2021-08-10'
author: Gabriel Stechschulte
layout: post
title: No Code, Dependency, and Building Technology
---

## Modernity and Abstraction

'Programmers', loosely speaking, in some form or another have always been developing software to automate tedious and repetitive tasks. Rightly so, as this is one of the tasks computers are designed to perform. As science and technology progresses, and gets more technological, there is a growing seperation between the maker and the user. This is one of the negative externalities of modernism - we enjoy the benefits of a more advanced and technologically adept society, but fewer and fewer people understand the inner workings. Andrej Karpathy has a jokingly short paragraph in his [blog](https://karpathy.github.io/2019/04/25/recipe/) on the matter, "A courageous developer has taken the burden of understanding query strings, urls, GET/POST requests, HTTP connections, and so on from you and largely hidden the complexity behind a few lines of code. This is what we are now familiar with and expect".

Ever since the rise of machine learning and data science, the industry was bound to develop no and or low code products for everything between data cleaning, processing, and annotation, to the implementation of models. This is an example of one of the highest forms of abstraction - you don't even need to be able to code. Knowledge of the problem at hand and some simple intuition into which model may provide a low validation error is almost all you need. A lower level form of abstraction is through the use of application programming interfaces (APIs) such as scikit-learn, PyTorch, etc. Using these APIs requires more technical skillsets, knowledge of first principles, and a deeper understanding of the problem than the no-code substitutes stated above. Lastly, we have the 'bare bones' implementation of algorithms - what most people usually call, "programming things from scratch". But even then, good luck writing your favorite model without using numpy, scipy, or JAX.

As teams of couragous developers and organizations seek to create products to help make everyday routines a commodity and more productive, it can be harder to learn technical methods from first principles as they have been abstracted away. There's no need, nor the time, to start and go writing everything from scratch, but having a solid intuition into what is going on under the hood can allow you to uniquely solve complex problems, debug more efficiently, contribute to open source software, etc.

## Switching Costs, Dependency and Open Source Goods

At least the two lower levels of abstractions usually rely on open-source software, whereas the no-code alternative is typically (at the moment) proprietary technology in the form of machine learning as a service and or platform. _[Edit 14.09.2021: H20.ai, HuggingFace, MakeML, CreateML, Google Cloud AutoML are all open-source services or platforms in the low or no code ML space]_ Open-source gives you the biggest flexibility that, if the space again changes, you can move things. Otherwise, you find yourself locked into a technology stack the way you were locked in to technologies from the ’80s and ’90s and 2000s.

Being locked into IT components can have serious opportunity costs. For example, switching from Mac to a Windows based PC involves not only the hardware costs of the computer itself, but also involves purchasing of a whole new library of software, and even more importantly, **learning** how to use a brand new system. When you, or an organization, decides to go the propriety route, these switching costs can be very high, and users may find themselves experiencing **lock-in**; a situation where the cost of changing to a different system is so high that switching is virtually inconcievable.

Of course, the producers of this hardware / sofware love the fact that you have an inelastic demand curve - a rise in prices won't affect demand much as switching costs are high. In summary, as machine learning platforms didn't really exist ~15 years ago, they will more than likely change quite a bit and you are dependent on the producer of continually adapting to industry trends and technological advancements while at the same time, staying flexible to cater to your edge cases when you need it.


## Leverage and Building Technology

It would be a full time job to stay present and up to date on everything being released in the space of "machine learning", but also to be knowledgeable of first princples and have skillsets to use the technologies is in another class of its own. Before AWS, APIs, open source, etc., as an organization or startup, it was likely the question, "we need amazing technical people who can build it _all_". Now, with the increase and rise of PaaS, SaaS, open source librarys and tooling, the question shifts from a question of "building", to a question of "leveraging existing tools". How long before no / low code gets good enough before we are asking, "why did we not build this with no-code tools?".

The highest form of leverage for a company is to develop and build out difficult and new technology. No-code, if developed right (and is ideally open-source), can still provide leverage and value-add, but any advantage just becomes table stakes. This is what will distinguish great teams vs. good teams; non-linear returns will continue to be through building out proprietery and difficult technology.
