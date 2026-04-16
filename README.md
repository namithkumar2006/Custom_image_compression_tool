# SVD Image Compression Tool — Linear Algebra Project

## Overview
A GUI application that compresses images using **Singular Value Decomposition (SVD)**,
a core technique in Linear Algebra.

## Mathematical Concepts Used
| Concept | Formula |
|---|---|
| SVD | A = U · Σ · Vᵀ |
| Low-rank approx | Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ |
| Frobenius error | ‖A − Aₖ‖_F = √(Σᵢ₌ₖ₊₁ σᵢ²) |
| PSNR | 10 · log₁₀(255² / MSE) dB |
| Energy retention | Σσᵢ² / Σσᵢ² × 100% |
| Compression ratio | (m·n) / k(m+n+1) |

## Features
- 📷 Side-by-side original vs compressed preview
- 📦 File size before & after (KB) with % reduction
- 🔢 SVD Rank slider (k) — controls quality vs compression
- 🎚️ JPEG Output Quality slider — controls saved file size
- 📉 PSNR / SSIM quality metrics
- ⚡ Energy retention & compression ratio display
- 📊 Three analysis plots:
  - Singular value spectrum
  - Cumulative energy curve
  - File size & PSNR vs k
- 💾 Save anywhere with format choice (JPG / PNG / BMP)

## Setup
```bash
pip install -r requirements.txt
python app.py
```

## How SVD Compression Works
1. Each colour channel (B, G, R) is treated as a matrix A ∈ ℝ^{m×n}
2. Full SVD is computed: A = U Σ Vᵀ
3. Only the top-k singular values are retained
4. Reconstruction: Aₖ = U[:, :k] · diag(Σ[:k]) · Vᵀ[:k, :]
5. By the Eckart–Young–Mirsky theorem this is the **optimal rank-k approximation**

## Requirements
- Python 3.10+
- numpy, opencv-python, Pillow, matplotlib
