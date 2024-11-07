# Neural SDF: Neural Implicit Signed Distance Fields (Work in Progress)

> ⚠️ **Note**: This project is currently under active development.

A JAX-based implementation for learning neural implicit representations of 3D geometries using signed distance fields (SDFs).

## Overview

This project provides tools to approximate the signed distance field of closed triangle meshes using neural networks. The core features include:

- Data:
  - Generates signed distance field data from arbitrary closed triangle meshes
  - Makes use of importance sampling (denser sampling around the surface)
- Model:
  - Introduces a new architecture which makes use of a 3D feature grid and positional encoding
  - The code is flexible and allows for easy experimentation with different architectures

## Installation


