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

- **Disentangled Knowledge Tracing**: Improves cross-domain KT accuracy and interpretability via explicit disentanglement of domain-shared and domain-exclusive knowledge components
- **Variational Attention Autoencoder (VAAE)**: Novel architecture that fuses hierarchical attention mechanisms with variational autoencoders for enhanced sequence modeling
- **Negative Transfer Reduction**: Advanced mutual information minimization technique enables precise knowledge transfer while reducing harmful negative transfer
- **Consistent Performance Gains**: Outperforms existing approaches across multiple CDKT scenarios, with average improvements of 3.09% AUC in course-level tasks
- **Multi-Domain Support**: Flexible framework designed for complex educational settings with multiple source and target domains

## Installation

```bash
git clone https://github.com/zenmezhemecaia/DisenKT.git
cd DisenKT
pip install -r requirements.txt
```
## Usage
```bash
Run the following command to start training the model:

python main.py
```
