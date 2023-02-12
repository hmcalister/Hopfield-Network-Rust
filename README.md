# Hopfield Network Rust
#### Author: Hayden McAlister

## Introduction

This is an implementation of the Hopfield network using Rust with nalgebra as the linear algebra backing. This project is intended to be clean and extensible, as well as blazing fast and scalable with CPU cores via threading. 

In future it may be interesting to try and port this project to use a different backend crate - one that leverages linear algebra on CUDA to scale instead with the GPU.


## Why Rust?

Rust was chosen for this project for the following reasons:

- It was found to be fast (see the profiling and testing [in this repository](https://github.com/hmcalister/Linear-Algebra-Profiling) - be sure to checkout the dashboard!)

- Tensorflow was found to scale much better by leveraging the GPU, but ensuring the code continued to scale required awkward vectorized methods that were prone to bugs.

- Go was found to scale very slightly better on the CPU, but after the [initial implementation](https://github.com/hmcalister/Hopfield-Network-Go) the language was found to be not quite strict enough. The lack of useful features like Enums or mapping over vectors made some methods clunky.

