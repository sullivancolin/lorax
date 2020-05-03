# Adapted From Joel Grus' Live Coding a Deep Learning Library

Original Repo [Here](https://github.com/joelgrus/joelnet)
Live Coding Video [Here](https://www.youtube.com/watch?v=o64FV-ez6Gw)

# Details of Reimplementation

* Replace numpy backend with [Jax](https://github.com/google/jax)
* Automatic calculation of gradients using Jax Autograd via `jax.grad`
* Allow for compiliation to GPU of TPU
* Seamlessly parallelize from single instance inference to batch inference with `jax.vmap`
* Additional activation layers and loss funcitons
* Plot Loss curves over time using Plotly
* Turn `train` into generator for early stopping and show progress bar.
* Jupyter Notebook and standalone scripts for both xor and fizzbuzz
* Implemented Dropout Layer
* Automatic Pytree class registration via inheritance
*