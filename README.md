# Adapted From Joel Grus' Live Coding a Deep Learning Library

Original Repo [Here](https://github.com/joelgrus/joelnet)
Live Coding Video [Here](https://www.youtube.com/watch?v=o64FV-ez6Gw)

# Details of Reimplementation

* Replace numpy backend with [Jax](https://github.com/google/jax)
* Automatic calculation of gradients using Jax Autograd via `jax.grad`
* Automatic Pytree class registration via inheritance
* Allow for compiliation to GPU or TPU
* layers are immutiable pydantic models with simple json definition
* Seamlessly parallelize from single instance inference to batch inference with `jax.vmap`
* Additional activation layers and loss funcitons
* Track progress with `wandb`
* Includes Dropout
* LSTM and Bidirectional LSTM
* Frozen Linear, Embedding, LSTMcell layers
* Experiment Config system with json schema compliant

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/niloch/colin_net/master)