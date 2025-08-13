# DisenKT: A Variational Attention-Based Approach for Disentangled Cross-Domain Knowledge Tracing

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper **"DisenKT: A Variational Attention-Based Approach for Disentangled Cross-Domain Knowledge Tracing"**

## Overview

DisenKT is a novel framework for cross-domain knowledge tracing that:
- Disentangles student knowledge states into domain-shared and domain-exclusive components
- Uses Variational Attention Autoencoders (VAAE) for effective sequence modeling
- Reduces negative transfer through mutual information minimization
- Outperforms existing approaches in both course-level and student-level CDKT scenarios

## Key Features

- 🧠 **Disentangled Representation Learning**: Separates transferable knowledge from domain-specific patterns
- ⏳ **Temporal Modeling**: Combines VAE with hierarchical attention for effective sequence modeling
- 🔄 **Cross-Domain Transfer**: Supports knowledge transfer across multiple educational domains
- 📊 **Interpretable Results**: Provides visualizations of knowledge state evolution

## Installation

```bash
git clone https://github.com/yourusername/DisenKT.git
cd DisenKT
pip install -r requirements.txt
