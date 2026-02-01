# FSD311 – Deep Compression

## Overview

This repository demonstrates the **Deep Compression** pipeline introduced in Song Han et al., _"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"_ applied to a LeNet model trained on the MNIST dataset. Deep Compression combines **pruning, weight quantization, and Huffman coding** to reduce model size while maintaining accuracy.

Additionally, a lightweight model has been trained for comparison, showing the performance of a similarly sized compressed model.


## Requirements

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Demo Notebook

The notebook [`LeNet.ipynb`](./notebooks/LeNet.ipynb) presents the main findings of this work, including:

* Application of the Deep Compression pipeline to LeNet
* Visualization of compression results
* Comparison between compressed and lightweight models

To save time, pre-trained models and encodings are provided in the following directories:

* `saves/` – trained model checkpoints
* `encodings/` – Huffman encodings

## Deep Compression PyTorch Pipeline

You can train and compress a new LeNet model using:

```bash
python main.py
```

This script runs the full Deep Compression pipeline.

 Note: Modifications will be necessary if you want to apply this pipeline to models trained on datasets other than MNIST.


## About

This implementation is based on the original Deep Compression repository by [mightydeveloper](https://github.com/mightydeveloper/Deep-Compression-PyTorch).

Some changes were made to:
* Fix existing bugs
* Automate the pipeline for LeNet