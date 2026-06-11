# 🎨 Generative AI Deep Dive: Diffusion Models, DCGANs & Stable Diffusion

A comprehensive collection of educational implementations demonstrating the foundations of modern Generative AI using PyTorch. This repository explores **Denoising Diffusion Probabilistic Models (DDPM)**, **Deep Convolutional Generative Adversarial Networks (DCGANs)**, and optimized **Stable Diffusion** pipelines for high-performance image generation.

---

## 📌 Overview

This project provides hands-on implementations of major generative modeling techniques used in modern AI systems. It covers the complete workflow from noise scheduling and denoising networks to adversarial training and state-of-the-art latent diffusion optimization techniques.

Whether you're learning Generative AI fundamentals or experimenting with image synthesis pipelines, this repository serves as a practical reference.

---

## 🚀 Features

### 🔹 Denoising Diffusion Probabilistic Models (DDPM)

* Linear beta noise scheduling.
* Forward diffusion (image corruption) process.
* Reverse denoising sampling process.
* U-Net based denoiser training.
* Image generation directly from Gaussian noise.

### 🔹 Deep Convolutional GAN (DCGAN)

* Generator and Discriminator implementations from scratch.
* Custom weight initialization.
* Label smoothing for stable training.
* CelebA dataset support.
* Epoch-wise image generation checkpoints.

### 🔹 Stable Diffusion Optimization

* FP16 mixed precision inference.
* Memory-efficient attention using xFormers.
* VAE slicing optimization.
* Reduced VRAM consumption.
* Fast text-to-image generation.

### 🔹 Advanced Generative AI Components

* Cross-Attention Mechanism.
* BERT Text Embeddings.
* Classifier-Free Guidance (CFG).
* Sinusoidal Time Embeddings.
* Latent Space Optimization Techniques.

---

## 🏗️ Architecture Overview

### 1️⃣ DDPM (Denoising Diffusion Probabilistic Model)

The diffusion process gradually adds Gaussian noise to an image over 1000 timesteps.

#### Forward Diffusion

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
```

Where:

* (x_0) = Original image
* (x_t) = Noisy image at timestep (t)
* (\epsilon) = Random Gaussian noise

#### Denoising Network

A U-Net architecture with:

* Attention Down Blocks
* Attention Up Blocks
* Skip Connections
* Multi-scale Feature Learning

Approximate model size:

* **28 Million Parameters**

---

### 2️⃣ DCGAN

#### Generator

* ConvTranspose2D Layers
* Batch Normalization
* ReLU Activations
* Final Tanh Activation

Output:

* 64 × 64 RGB Images

#### Discriminator

* Strided Convolutions
* Batch Normalization
* LeakyReLU (0.2)
* Sigmoid Output Layer

Output:

* Real/Fake Probability Score

---

### 3️⃣ Stable Diffusion Optimization

#### xFormers Attention

Benefits:

* Reduced memory usage
* Faster attention computation
* Scalable image generation

#### Automatic Mixed Precision (AMP)

Advantages:

* FP16 inference
* Lower VRAM requirements
* Faster execution speeds
* Improved throughput

---

## 📂 Project Structure

```text
├── DDPM/
│   ├── forward_diffusion.py
│   ├── reverse_sampling.py
│   └── unet_training.py
│
├── DCGAN/
│   ├── generator.py
│   ├── discriminator.py
│   └── train_dcgan.py
│
├── Stable_Diffusion/
│   ├── optimized_pipeline.py
│   └── benchmark.py
│
├── outputs/
│   ├── forward_diffusion.png
│   ├── generated_samples/
│   └── gan_epoch_images/
│
└── README.md
```

---

## 📦 Installation

Install all required dependencies:

```bash
pip install torch torchvision matplotlib pillow
pip install diffusers transformers xformers
```

---

## ▶️ Running the Project

### DDPM Training & Sampling

```python
# Train denoising network
python unet_training.py

# Generate samples
python reverse_sampling.py
```

Expected Outputs:

* Forward diffusion visualization
* Reverse denoising progression
* Generated image samples

---

### DCGAN Training

```python
python train_dcgan.py
```

Typical Training Range:

```text
Loss_D: 0.4 - 0.7
Loss_G: 0.5 - 1.5
```

Outputs:

* Generated face images
* Epoch checkpoints
* Training progress logs

---

### Stable Diffusion Benchmark

```python
python benchmark.py
```

Guidance Scale Recommendations:

| CFG Scale | Behavior                               |
| --------- | -------------------------------------- |
| 1.0       | High creativity, weak prompt adherence |
| 7.5       | Balanced and recommended               |
| 10.0      | Strong prompt alignment                |
| 15.0      | Over-processed outputs and artifacts   |

---

## 📊 Performance Benchmarks

Optimized Stable Diffusion Performance
(FP16 + xFormers)

| Metric                  | Value             |
| ----------------------- | ----------------- |
| Average Generation Time | ~1.83 sec/image   |
| Throughput              | ~0.546 images/sec |
| VRAM Allocated          | ~6.3 GB           |
| VRAM Reserved           | ~7.2 GB           |
| Inference Steps         | 30                |

### Key Observation

Using FP16, VAE Slicing, and xFormers dramatically reduces memory consumption while maintaining high-quality image generation performance.

---

## 🎯 Learning Outcomes

This repository demonstrates:

* Diffusion Model Fundamentals
* GAN Training Strategies
* U-Net Architectures
* Attention Mechanisms
* Latent Diffusion Techniques
* Memory Optimization for Generative AI
* Text-to-Image Generation Workflows
* Production-Oriented AI Inference

---

## 📚 References

* DDPM (Denoising Diffusion Probabilistic Models)
* Stable Diffusion Research Papers
* Hugging Face Diffusers Library
* PyTorch Documentation
* DCGAN Paper (Radford et al.)

---

## 🙏 Acknowledgments

Educational examples and implementation concepts are inspired by the NVIDIA Internship Training Series and widely adopted Generative AI research literature.

---

## 📄 License

This project is released under the MIT License and is intended for educational, research, and learning purposes.

