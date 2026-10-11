# Image Noise Reduction Using Convolution

[![Course](https://img.shields.io/badge/Course-Deep%20Learning%20Essentials-blue.svg)](https://github.com)
[![Course Code](https://img.shields.io/badge/Course%20Code-21ML53T--26-orange.svg)](https://github.com)
[![Assignment](https://img.shields.io/badge/Assignment-Innovative%20Assignment%20II-green.svg)](https://github.com)
[![Topic](https://img.shields.io/badge/Topic-CO3%20%7C%20Level%20K3-purple.svg)](https://github.com)

**Vel Tech High Tech Dr. Rangarajan Dr. Sakunthala Engineering College**  
Department of Computer Science Engineering (Artificial Intelligence and Machine Learning)  
*Year / Semester:* III / V  

---

## 📌 Problem Statement
> **Topic:** Image Noise Reduction Using Convolution  
> **Course Outcome:** CO3 | **Bloom's Taxonomy Level:** K3  
> **Description:** Develop a program that reduces noise from digital images using average and Gaussian-style convolution filters and evaluates the output.

---

## 📖 Theoretical Concepts

### 1. 2D Spatial Convolution
Discrete 2D convolution transforms an input image $I(x, y)$ using a spatial kernel $K(m, n)$ of dimensions $(2a+1) \times (2b+1)$:
$$ (I * K)(x, y) = \sum_{m=-a}^{a} \sum_{n=-b}^{b} I(x - m, y - n) \cdot K(m, n) $$
In this project, custom 2D convolution is implemented from scratch using NumPy with reflection padding to prevent border darkening artifacts.

### 2. Average (Box / Mean) Convolution Filter
Assigns equal uniform weight to all neighborhood pixels:
$$ K_{\text{avg}} = \frac{1}{k^2} \begin{bmatrix} 1 & 1 & \dots & 1 \\ 1 & 1 & \dots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \dots & 1 \end{bmatrix}_{k \times k} $$
- **Characteristics:** Fast and simple; reduces high-frequency variance but causes severe edge blur and loss of sharp boundaries.

### 3. Gaussian-Style Convolution Filter
Weights are distributed according to a 2D isotropic Gaussian bell curve centered at the origin:
$$ G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right) $$
- **Characteristics:** Preserves edge sharpness significantly better than box filtering because closer pixels contribute more strongly to the central output than distant pixels.

### 4. Noise Models
- **Additive White Gaussian Noise (AWGN):** Models thermal and electronic sensor noise ($\mu = 0, \sigma = 25$).
- **Salt-and-Pepper Noise:** Models transmission errors and faulty sensor photodiode spikes (5% randomized impulse noise).

### 5. Quantitative Evaluation Metrics
- **Mean Squared Error (MSE):**
  $$ \text{MSE} = \frac{1}{M \cdot N} \sum_{x=1}^{M}\sum_{y=1}^{N} (I_{\text{clean}}(x, y) - I_{\text{denoised}}(x, y))^2 $$
- **Peak Signal-to-Noise Ratio (PSNR) in dB:**
  $$ \text{PSNR} = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) $$
- **Structural Similarity Index (SSIM):** Measures luminance, contrast, and structural correlation in $[-1, 1]$.

---

## 📁 Repository Structure
```
image-noise-reduction-convolution/
├── README.md                          # Project documentation and guide
├── requirements.txt                   # Python dependencies
├── run_project.bat                    # One-click execution script for Windows
├── dataset/
│   ├── sample_image.png               # High-contrast geometric benchmark image
│   ├── natural_scene.png              # Natural scene photograph
│   └── generate_dataset.py            # Synthetic dataset generator script
├── src/
│   └── image_noise_reduction.py       # Core Python program (from scratch + fast)
├── notebooks/
│   └── Noise_Reduction_Analysis.ipynb # Step-by-step interactive Jupyter notebook
├── results/
│   ├── denoising_comparison.png       # Side-by-side comparison grid
│   ├── kernel_size_tradeoff.png       # Trade-off across 3x3, 5x5, 7x7 kernels
│   ├── metrics_comparison_chart.png   # Bar plots of PSNR, MSE, and SSIM
│   └── metrics_summary.csv            # Quantitative evaluation data table
└── screenshots/
    └── output.png                     # Master dashboard screenshot for submission
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your computer.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nishant17s/Image-Noise-Reduction-Using-Convolution_DLE.git
cd Image-Noise-Reduction-Using-Convolution_DLE
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### Method 1: One-Click Execution (Windows)
Double-click `run_project.bat` or run:
```cmd
run_project.bat
```

### Method 2: Command Line
Run the core pipeline on the default benchmark image:
```bash
python src/image_noise_reduction.py
```

Run on a custom image (e.g. natural scene photo):
```bash
python src/image_noise_reduction.py --image dataset/natural_scene.png
```

### Method 3: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/Noise_Reduction_Analysis.ipynb
```

---

## 📊 Experimental Results & Evaluation

The table below summarizes the quantitative evaluation on the 512×512 benchmark image corrupted with Gaussian noise ($\sigma=25$):

| Noise Type | Filter Applied | Kernel Size | MSE (Lower is Better) | PSNR (dB) (Higher is Better) | SSIM (Higher is Better) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gaussian ($\sigma=25$)** | **Corrupted Baseline** | **N/A** | **531.34** | **20.88 dB** | **0.3003** |
| Gaussian ($\sigma=25$) | Average Filter | 3×3 | 475.39 | 21.36 dB | 0.6135 |
| Gaussian ($\sigma=25$) | **Gaussian Filter** | **3×3** | **376.71** | **22.37 dB** | **0.6101** |
| Gaussian ($\sigma=25$) | Average Filter | 5×5 | 685.32 | 19.77 dB | 0.7419 |
| Gaussian ($\sigma=25$) | Gaussian Filter | 5×5 | 562.69 | 20.63 dB | 0.7546 |
| Gaussian ($\sigma=25$) | Average Filter | 7×7 | 884.12 | 18.67 dB | 0.7615 |
| Gaussian ($\sigma=25$) | Gaussian Filter | 7×7 | 738.61 | 19.45 dB | 0.7888 |

### Key Observations:
1. **Gaussian Filter Advantage:** The $3\times 3$ Gaussian filter outperforms the $3\times 3$ Average filter by **+1.01 dB in PSNR** and exhibits lower MSE (376.71 vs 475.39).
2. **Smoothing vs. Blurring Trade-off:** While larger kernels ($5\times 5$, $7\times 7$) achieve higher structural noise suppression in flat regions (raising SSIM up to 0.7888), they introduce significant blurring across high-frequency boundaries.
3. **Optimal Filter Selection:** For additive Gaussian noise, the $3\times 3$ Gaussian filter strikes the best balance between noise reduction and edge retention.

---

## 🎯 Viva Demonstration & Live Modification Tips
- **Where is the custom convolution implemented?**  
  In `src/image_noise_reduction.py` -> `convolve2d_single_channel` and `convolve_scratch`.
- **How to demonstrate custom convolution equivalence?**  
  Run `python src/image_noise_reduction.py`; Step 1 verifies that scratch convolution matches reference convolution mathematically with MSE < 1.0.
- **How to modify kernel size live?**  
  Pass different sizes in `kernel_sizes = [3, 5, 7, 9]` in `src/image_noise_reduction.py`.

---

## 📜 Academic Integrity & Acknowledgements
- Developed individually as part of **Innovative Assignment – II** for **Deep Learning Essentials (21ML53T-26)**.
- Formulated using standard scientific Python libraries: NumPy, Matplotlib, OpenCV, and SciPy.
