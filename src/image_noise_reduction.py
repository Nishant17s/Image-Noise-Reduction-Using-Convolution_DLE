"""
Image Noise Reduction Using Convolution
Department of Computer Science Engineering (AI & ML)
Course: Deep Learning Essentials (21ML53T-26)
Topic: Image Noise Reduction Using Convolution Filters (CO3 - K3)

This module implements:
1. Custom 2D spatial convolution from scratch using pure NumPy.
2. Average (Box / Mean) convolution filter kernel generation.
3. Gaussian-style convolution filter kernel generation.
4. Noise degradation models (Additive White Gaussian Noise and Salt-and-Pepper Noise).
5. Quantitative evaluation metrics: Mean Squared Error (MSE),
   Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
6. Comparative visual and tabular evaluation with automatic graph generation.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.ndimage import convolve as scipy_convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# 1. IMAGE I/O UTILITY
# ============================================================================

def load_rgb_image(image_path: str) -> np.ndarray:
    """
    Loads an image file into an RGB NumPy array (uint8).
    Supports OpenCV and PIL backends transparently.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if HAS_CV2:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"OpenCV failed to decode image: {image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        pil_img = Image.open(image_path).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)


# ============================================================================
# 2. NOISE SIMULATION FUNCTIONS
# ============================================================================

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to an image.
    
    Formula:
        I_noisy(x, y) = clip(I_clean(x, y) + N(mean, sigma^2), 0, 255)
    
    Args:
        image: Clean input image (uint8, values in [0, 255]).
        mean: Mean of the Gaussian noise distribution (default 0.0).
        sigma: Standard deviation representing noise intensity (default 25.0).
        
    Returns:
        Corrupted image as uint8 array in [0, 255].
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.04, s_vs_p: float = 0.5) -> np.ndarray:
    """
    Adds Salt-and-Pepper (impulse) noise to an image.
    
    Args:
        image: Clean input image (uint8).
        amount: Total percentage of corrupted pixels (e.g. 0.04 for 4%).
        s_vs_p: Ratio of salt (white) vs pepper (black) pixels (default 0.5).
        
    Returns:
        Corrupted image as uint8 array.
    """
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num_total = int(h * w * amount)
    
    # Salt noise (white pixels = 255)
    num_salt = int(num_total * s_vs_p)
    salt_coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise (black pixels = 0)
    num_pepper = int(num_total * (1.0 - s_vs_p))
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy


# ============================================================================
# 3. FILTER KERNEL GENERATORS
# ============================================================================

def create_average_kernel(kernel_size: int = 3) -> np.ndarray:
    """
    Creates an Average (Box / Mean) convolution filter kernel.
    Every element has equal weight: 1 / (kernel_size^2).
    
    Formula:
        K(m, n) = 1 / (kernel_size^2)  for all m, n in [0, kernel_size-1]
        
    Args:
        kernel_size: Dimension of the square kernel (must be odd: 3, 5, 7, ...).
        
    Returns:
        2D float64 kernel summing to 1.0.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer (3, 5, 7, ...).")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    return kernel / (kernel_size * kernel_size)


def create_gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Creates a 2D Gaussian convolution filter kernel.
    
    Formula:
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
        Normalized so sum(G) = 1.0.
        
    Args:
        kernel_size: Size of the square kernel (must be odd: 3, 5, 7, ...).
        sigma: Standard deviation controlling the spread of the bell curve.
        
    Returns:
        2D float64 kernel normalized such that all elements sum to 1.0.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer (3, 5, 7, ...).")
    
    radius = kernel_size // 2
    # Generate discrete spatial coordinate grids centered at (0, 0)
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    
    # Gaussian distribution equation
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # Normalize weights so sum equals 1.0 (preserves image average brightness)
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel / kernel_sum
        
    return kernel


# ============================================================================
# 4. SPATIAL 2D CONVOLUTION (FROM SCRATCH & FAST)
# ============================================================================

def convolve2d_single_channel(image_channel: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect') -> np.ndarray:
    """
    Performs 2D discrete convolution from scratch on a single image channel.
    
    Mathematical formula:
        Output(i, j) = sum_{m=-a}^{a} sum_{n=-b}^{b} PaddedImage(i - m, j - n) * Kernel(m, n)
        
    Reflection padding is used to prevent dark border artifacts.
    
    Args:
        image_channel: 2D numpy array.
        kernel: 2D numpy array of filter weights.
        padding_mode: 'reflect' (recommended) or 'constant'.
        
    Returns:
        2D float64 array of convolved pixels.
    """
    img = image_channel.astype(np.float64)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Boundary padding
    if padding_mode == 'reflect':
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float64)
    
    # Flip kernel for formal mathematical convolution
    flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    
    # Sliding window discrete convolution
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(window * flipped_kernel)
            
    return output


def convolve_scratch(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies custom 2D convolution from scratch to 2D grayscale or 3D RGB images.
    
    Args:
        image: 2D grayscale or 3D RGB numpy array.
        kernel: 2D convolution kernel.
        
    Returns:
        Filtered image as uint8 array in [0, 255].
    """
    if image.ndim == 2:
        res = convolve2d_single_channel(image, kernel)
        return np.clip(res, 0, 255).astype(np.uint8)
    elif image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            filtered_c = convolve2d_single_channel(image[:, :, c], kernel)
            channels.append(filtered_c)
        stacked = np.stack(channels, axis=2)
        return np.clip(stacked, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def convolve_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast convolution execution engine with automatic fallback:
    - Uses OpenCV filter2D if available.
    - Uses SciPy ndimage convolve if OpenCV is absent.
    - Uses custom convolve_scratch if neither is available.
    """
    if HAS_CV2:
        return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    elif HAS_SCIPY:
        if image.ndim == 2:
            res = scipy_convolve(image.astype(np.float64), kernel, mode='reflect')
            return np.clip(res, 0, 255).astype(np.uint8)
        else:
            channels = [scipy_convolve(image[:, :, c].astype(np.float64), kernel, mode='reflect') for c in range(image.shape[2])]
            return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)
    else:
        return convolve_scratch(image, kernel)


# ============================================================================
# 5. QUANTITATIVE EVALUATION METRICS
# ============================================================================

def compute_mse(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """
    Computes Mean Squared Error (MSE) between reference and evaluated images.
    
    Formula:
        MSE = (1 / (M * N)) * sum_{x=1}^{M} sum_{y=1}^{N} (I_true(x, y) - I_pred(x, y))^2
    
    Interpretation:
        Lower MSE indicates smaller cumulative squared error and higher reconstruction accuracy.
    """
    true_f = img_true.astype(np.float64)
    pred_f = img_pred.astype(np.float64)
    return float(np.mean((true_f - pred_f) ** 2))


def compute_psnr(img_true: np.ndarray, img_pred: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))
        
    Interpretation:
        Higher PSNR indicates superior noise suppression and greater image fidelity.
        Values between 20-30 dB represent acceptable quality; >30 dB is high quality.
    """
    mse = compute_mse(img_true, img_pred)
    if mse == 0.0:
        return float('inf')
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) for a single channel.
    
    Formula:
        SSIM(x, y) = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / 
                     ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
                     
    Evaluates:
        1. Luminance comparison
        2. Contrast comparison
        3. Structural correlation
        
    Range: [-1.0, 1.0], where 1.0 signifies identical structural information.
    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 11x11 Gaussian window (standard Wang et al.)
    kernel = create_gaussian_kernel(kernel_size=11, sigma=1.5)

    if HAS_CV2:
        mu1 = cv2.filter2D(img1, -1, kernel, borderType=cv2.BORDER_REFLECT)
        mu2 = cv2.filter2D(img2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    elif HAS_SCIPY:
        mu1 = scipy_convolve(img1, kernel, mode='reflect')
        mu2 = scipy_convolve(img2, kernel, mode='reflect')
    else:
        mu1 = convolve2d_single_channel(img1, kernel)
        mu2 = convolve2d_single_channel(img2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    if HAS_CV2:
        sigma1_sq = cv2.filter2D(img1 * img1, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu1_sq
        sigma2_sq = cv2.filter2D(img2 * img2, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu1_mu2
    elif HAS_SCIPY:
        sigma1_sq = scipy_convolve(img1 * img1, kernel, mode='reflect') - mu1_sq
        sigma2_sq = scipy_convolve(img2 * img2, kernel, mode='reflect') - mu2_sq
        sigma12 = scipy_convolve(img1 * img2, kernel, mode='reflect') - mu1_mu2
    else:
        sigma1_sq = convolve2d_single_channel(img1 * img1, kernel) - mu1_sq
        sigma2_sq = convolve2d_single_channel(img2 * img2, kernel) - mu2_sq
        sigma12 = convolve2d_single_channel(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def compute_ssim(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """
    Computes Mean SSIM across 2D grayscale or 3D RGB channels.
    """
    if img_true.ndim == 2:
        return compute_ssim_channel(img_true, img_pred)
    elif img_true.ndim == 3:
        ssims = [compute_ssim_channel(img_true[:, :, c], img_pred[:, :, c]) for c in range(img_true.shape[2])]
        return float(np.mean(ssims))
    return 0.0


def evaluate_image_quality(clean_img: np.ndarray, processed_img: np.ndarray) -> dict:
    """
    Evaluates MSE, PSNR, and SSIM metrics.
    """
    mse = compute_mse(clean_img, processed_img)
    psnr = compute_psnr(clean_img, processed_img)
    ssim = compute_ssim(clean_img, processed_img)
    return {
        "MSE": round(mse, 2),
        "PSNR (dB)": round(psnr, 2),
        "SSIM": round(ssim, 4)
    }


# ============================================================================
# 6. EXPERIMENTAL STUDY & VISUAL PIPELINE
# ============================================================================

def run_noise_reduction_experiment(
    image_path: str,
    output_dir: str = "results",
    screenshots_dir: str = "screenshots"
):
    """
    Runs the complete evaluation:
    1. Loads clean reference image.
    2. Tests scratch 2D convolution and validates mathematical equivalence.
    3. Injects synthetic Gaussian noise (sigma=25) and Salt-and-Pepper noise (5%).
    4. Convolves with Average and Gaussian filters across multiple kernel sizes (3x3, 5x5, 7x7).
    5. Computes MSE, PSNR, SSIM metrics.
    6. Generates and saves high-resolution publication-quality plots and summary tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    print("=" * 75)
    print("      IMAGE NOISE REDUCTION USING CONVOLUTION EXPERIMENT")
    print("      Department of CSE (AI & ML) - Course: 21ML53T-26")
    print("=" * 75)
    print(f"Loading reference image: {image_path}")
    
    clean_rgb = load_rgb_image(image_path)
    h, w, c = clean_rgb.shape
    print(f"Loaded image resolution: {w}x{h} ({c} channels)")
    
    # ------------------------------------------------------------------------
    # STEP 1: VERIFY SCRATCH CONVOLUTION IMPLEMENTATION
    # ------------------------------------------------------------------------
    print("\n[Step 1] Verifying custom 2D convolution implementation...")
    # Test on a 128x128 patch
    test_patch = clean_rgb[:128, :128].copy()
    test_kernel = create_gaussian_kernel(3, sigma=1.0)
    
    t0 = time.time()
    scratch_out = convolve_scratch(test_patch, test_kernel)
    t_scratch = time.time() - t0
    
    t0 = time.time()
    fast_out = convolve_fast(test_patch, test_kernel)
    t_fast = time.time() - t0
    
    diff_mse = compute_mse(scratch_out, fast_out)
    print(f"   * Scratch Convolution Execution Time: {t_scratch:.4f}s")
    print(f"   * Fast Convolution Execution Time:    {t_fast:.4f}s")
    print(f"   * Difference MSE (Scratch vs Fast):   {diff_mse:.6f}")
    if diff_mse < 1.0:
        print("   -> VERIFICATION PASSED: Scratch convolution matches reference mathematically.")
    
    # ------------------------------------------------------------------------
    # STEP 2: CORRUPT IMAGE WITH NOISE
    # ------------------------------------------------------------------------
    print("\n[Step 2] Corrupting digital image with synthetic noise models...")
    np.random.seed(42)  # Fixed seed for reproducibility
    noisy_gaussian = add_gaussian_noise(clean_rgb, mean=0.0, sigma=25.0)
    noisy_sp = add_salt_and_pepper_noise(clean_rgb, amount=0.05)
    
    base_metrics_gauss = evaluate_image_quality(clean_rgb, noisy_gaussian)
    base_metrics_sp = evaluate_image_quality(clean_rgb, noisy_sp)
    
    print(f"   * Gaussian Noisy Baseline:       MSE={base_metrics_gauss['MSE']}, PSNR={base_metrics_gauss['PSNR (dB)']} dB, SSIM={base_metrics_gauss['SSIM']}")
    print(f"   * Salt & Pepper Noisy Baseline:  MSE={base_metrics_sp['MSE']}, PSNR={base_metrics_sp['PSNR (dB)']} dB, SSIM={base_metrics_sp['SSIM']}")

    # ------------------------------------------------------------------------
    # STEP 3: APPLY AVERAGE & GAUSSIAN CONVOLUTION FILTERS
    # ------------------------------------------------------------------------
    print("\n[Step 3] Applying Average and Gaussian Convolution Filters...")
    kernel_sizes = [3, 5, 7]
    
    results_records = []
    
    # Baseline records
    results_records.append({
        "Noise Type": "Gaussian (sigma=25)",
        "Filter Applied": "None (Corrupted Baseline)",
        "Kernel Size": "N/A",
        "MSE": base_metrics_gauss["MSE"],
        "PSNR (dB)": base_metrics_gauss["PSNR (dB)"],
        "SSIM": base_metrics_gauss["SSIM"]
    })
    
    filtered_images = {
        "Gaussian Noise": {
            "Original": clean_rgb,
            "Noisy": noisy_gaussian,
            "Avg_3": None, "Avg_5": None, "Avg_7": None,
            "Gauss_3": None, "Gauss_5": None, "Gauss_7": None
        },
        "Salt & Pepper Noise": {
            "Original": clean_rgb,
            "Noisy": noisy_sp,
            "Avg_3": None, "Avg_5": None, "Avg_7": None,
            "Gauss_3": None, "Gauss_5": None, "Gauss_7": None
        }
    }
    
    for k in kernel_sizes:
        k_avg = create_average_kernel(k)
        sigma_val = float(k) / 3.0
        k_gauss = create_gaussian_kernel(k, sigma=sigma_val)
        
        # 1. Gaussian Noise Filtering
        avg_filtered_g = convolve_fast(noisy_gaussian, k_avg)
        gauss_filtered_g = convolve_fast(noisy_gaussian, k_gauss)
        filtered_images["Gaussian Noise"][f"Avg_{k}"] = avg_filtered_g
        filtered_images["Gaussian Noise"][f"Gauss_{k}"] = gauss_filtered_g
        
        m_avg_g = evaluate_image_quality(clean_rgb, avg_filtered_g)
        m_gauss_g = evaluate_image_quality(clean_rgb, gauss_filtered_g)
        
        results_records.append({
            "Noise Type": "Gaussian (sigma=25)",
            "Filter Applied": "Average Filter",
            "Kernel Size": f"{k}x{k}",
            "MSE": m_avg_g["MSE"],
            "PSNR (dB)": m_avg_g["PSNR (dB)"],
            "SSIM": m_avg_g["SSIM"]
        })
        results_records.append({
            "Noise Type": "Gaussian (sigma=25)",
            "Filter Applied": "Gaussian Filter",
            "Kernel Size": f"{k}x{k}",
            "MSE": m_gauss_g["MSE"],
            "PSNR (dB)": m_gauss_g["PSNR (dB)"],
            "SSIM": m_gauss_g["SSIM"]
        })

        # 2. Salt & Pepper Noise Filtering
        avg_filtered_sp = convolve_fast(noisy_sp, k_avg)
        gauss_filtered_sp = convolve_fast(noisy_sp, k_gauss)
        filtered_images["Salt & Pepper Noise"][f"Avg_{k}"] = avg_filtered_sp
        filtered_images["Salt & Pepper Noise"][f"Gauss_{k}"] = gauss_filtered_sp
        
        m_avg_sp = evaluate_image_quality(clean_rgb, avg_filtered_sp)
        m_gauss_sp = evaluate_image_quality(clean_rgb, gauss_filtered_sp)
        
        results_records.append({
            "Noise Type": "Salt & Pepper (5%)",
            "Filter Applied": "Average Filter",
            "Kernel Size": f"{k}x{k}",
            "MSE": m_avg_sp["MSE"],
            "PSNR (dB)": m_avg_sp["PSNR (dB)"],
            "SSIM": m_avg_sp["SSIM"]
        })
        results_records.append({
            "Noise Type": "Salt & Pepper (5%)",
            "Filter Applied": "Gaussian Filter",
            "Kernel Size": f"{k}x{k}",
            "MSE": m_gauss_sp["MSE"],
            "PSNR (dB)": m_gauss_sp["PSNR (dB)"],
            "SSIM": m_gauss_sp["SSIM"]
        })

    # Save Metrics CSV
    df_metrics = pd.DataFrame(results_records)
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[Step 4] Metrics summary saved to: {csv_path}")
    print(df_metrics.to_string(index=False))

    # ------------------------------------------------------------------------
    # STEP 5: VISUALIZATION 1 - COMPREHENSIVE FILTER COMPARISON GRID
    # ------------------------------------------------------------------------
    print("\n[Step 5] Generating Comparison Visualizations...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    
    # Row 1: Gaussian Noise Filtering
    axes[0, 0].imshow(clean_rgb)
    axes[0, 0].set_title("Ground Truth (Clean)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(noisy_gaussian)
    axes[0, 1].set_title(f"Gaussian Noisy\nPSNR: {base_metrics_gauss['PSNR (dB)']} dB | SSIM: {base_metrics_gauss['SSIM']}", fontsize=10)
    axes[0, 1].axis("off")
    
    m_a3 = evaluate_image_quality(clean_rgb, filtered_images["Gaussian Noise"]["Avg_3"])
    axes[0, 2].imshow(filtered_images["Gaussian Noise"]["Avg_3"])
    axes[0, 2].set_title(f"Average Filter (3x3)\nPSNR: {m_a3['PSNR (dB)']} dB | SSIM: {m_a3['SSIM']}", fontsize=10)
    axes[0, 2].axis("off")
    
    m_g3 = evaluate_image_quality(clean_rgb, filtered_images["Gaussian Noise"]["Gauss_3"])
    axes[0, 3].imshow(filtered_images["Gaussian Noise"]["Gauss_3"])
    axes[0, 3].set_title(f"Gaussian Filter (3x3)\nPSNR: {m_g3['PSNR (dB)']} dB | SSIM: {m_g3['SSIM']}", fontsize=10, color="darkgreen", fontweight="bold")
    axes[0, 3].axis("off")

    # Row 2: Salt & Pepper Noise Filtering
    axes[1, 0].imshow(clean_rgb)
    axes[1, 0].set_title("Ground Truth (Clean)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(noisy_sp)
    axes[1, 1].set_title(f"Salt & Pepper Noisy\nPSNR: {base_metrics_sp['PSNR (dB)']} dB | SSIM: {base_metrics_sp['SSIM']}", fontsize=10)
    axes[1, 1].axis("off")
    
    m_a3_sp = evaluate_image_quality(clean_rgb, filtered_images["Salt & Pepper Noise"]["Avg_3"])
    axes[1, 2].imshow(filtered_images["Salt & Pepper Noise"]["Avg_3"])
    axes[1, 2].set_title(f"Average Filter (3x3)\nPSNR: {m_a3_sp['PSNR (dB)']} dB | SSIM: {m_a3_sp['SSIM']}", fontsize=10)
    axes[1, 2].axis("off")
    
    m_g3_sp = evaluate_image_quality(clean_rgb, filtered_images["Salt & Pepper Noise"]["Gauss_3"])
    axes[1, 3].imshow(filtered_images["Salt & Pepper Noise"]["Gauss_3"])
    axes[1, 3].set_title(f"Gaussian Filter (3x3)\nPSNR: {m_g3_sp['PSNR (dB)']} dB | SSIM: {m_g3_sp['SSIM']}", fontsize=10, color="darkgreen", fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle("Image Noise Reduction using Convolution (Average vs Gaussian)", fontsize=16, fontweight="bold", y=0.98)
    fig_comp_path = os.path.join(output_dir, "denoising_comparison.png")
    plt.savefig(fig_comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   * Comparison grid saved: {fig_comp_path}")

    # ------------------------------------------------------------------------
    # STEP 6: VISUALIZATION 2 - KERNEL SIZE TRADEOFF ANALYSIS
    # ------------------------------------------------------------------------
    fig_k, axes_k = plt.subplots(2, 3, figsize=(15, 9))
    for idx, k in enumerate([3, 5, 7]):
        img_avg = filtered_images["Gaussian Noise"][f"Avg_{k}"]
        img_g = filtered_images["Gaussian Noise"][f"Gauss_{k}"]
        
        m_a = evaluate_image_quality(clean_rgb, img_avg)
        m_g = evaluate_image_quality(clean_rgb, img_g)
        
        axes_k[0, idx].imshow(img_avg)
        axes_k[0, idx].set_title(f"Average Filter ({k}x{k})\nPSNR: {m_a['PSNR (dB)']} dB | SSIM: {m_a['SSIM']}", fontsize=11)
        axes_k[0, idx].axis("off")
        
        axes_k[1, idx].imshow(img_g)
        axes_k[1, idx].set_title(f"Gaussian Filter ({k}x{k})\nPSNR: {m_g['PSNR (dB)']} dB | SSIM: {m_g['SSIM']}", fontsize=11)
        axes_k[1, idx].axis("off")
        
    axes_k[0, 0].set_ylabel("Average Filter", fontsize=12, fontweight="bold")
    axes_k[1, 0].set_ylabel("Gaussian Filter", fontsize=12, fontweight="bold")
    plt.suptitle("Kernel Size Trade-off: Noise Smoothing vs. Edge Blurring (3x3, 5x5, 7x7)", fontsize=15, fontweight="bold")
    fig_k_path = os.path.join(output_dir, "kernel_size_tradeoff.png")
    plt.savefig(fig_k_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   * Kernel size tradeoff plot saved: {fig_k_path}")

    # ------------------------------------------------------------------------
    # STEP 7: VISUALIZATION 3 - QUANTITATIVE METRIC CHARTS
    # ------------------------------------------------------------------------
    df_gauss = df_metrics[(df_metrics["Noise Type"].str.contains("Gaussian")) & (df_metrics["Kernel Size"] != "N/A")]
    
    fig_metrics, (ax_psnr, ax_ssim, ax_mse) = plt.subplots(1, 3, figsize=(16, 5))
    
    k_labels = ["3x3", "5x5", "7x7"]
    x = np.arange(len(k_labels))
    width = 0.35
    
    avg_psnr = df_gauss[df_gauss["Filter Applied"] == "Average Filter"]["PSNR (dB)"].values
    gauss_psnr = df_gauss[df_gauss["Filter Applied"] == "Gaussian Filter"]["PSNR (dB)"].values
    
    ax_psnr.bar(x - width/2, avg_psnr, width, label="Average Filter", color="#3498db")
    ax_psnr.bar(x + width/2, gauss_psnr, width, label="Gaussian Filter", color="#2ecc71")
    ax_psnr.axhline(base_metrics_gauss["PSNR (dB)"], color="red", linestyle="--", label=f"Noisy Baseline ({base_metrics_gauss['PSNR (dB)']} dB)")
    ax_psnr.set_ylabel("PSNR (dB) - Higher is Better", fontsize=11)
    ax_psnr.set_title("Peak Signal-to-Noise Ratio (PSNR)", fontsize=12, fontweight="bold")
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(k_labels)
    ax_psnr.legend(loc="lower right")
    ax_psnr.grid(axis='y', linestyle=':', alpha=0.6)
    
    avg_ssim = df_gauss[df_gauss["Filter Applied"] == "Average Filter"]["SSIM"].values
    gauss_ssim = df_gauss[df_gauss["Filter Applied"] == "Gaussian Filter"]["SSIM"].values
    
    ax_ssim.bar(x - width/2, avg_ssim, width, label="Average Filter", color="#3498db")
    ax_ssim.bar(x + width/2, gauss_ssim, width, label="Gaussian Filter", color="#2ecc71")
    ax_ssim.axhline(base_metrics_gauss["SSIM"], color="red", linestyle="--", label=f"Noisy Baseline ({base_metrics_gauss['SSIM']})")
    ax_ssim.set_ylabel("SSIM - Higher is Better", fontsize=11)
    ax_ssim.set_title("Structural Similarity Index (SSIM)", fontsize=12, fontweight="bold")
    ax_ssim.set_xticks(x)
    ax_ssim.set_xticklabels(k_labels)
    ax_ssim.legend(loc="lower right")
    ax_ssim.grid(axis='y', linestyle=':', alpha=0.6)

    avg_mse = df_gauss[df_gauss["Filter Applied"] == "Average Filter"]["MSE"].values
    gauss_mse = df_gauss[df_gauss["Filter Applied"] == "Gaussian Filter"]["MSE"].values
    
    ax_mse.bar(x - width/2, avg_mse, width, label="Average Filter", color="#3498db")
    ax_mse.bar(x + width/2, gauss_mse, width, label="Gaussian Filter", color="#2ecc71")
    ax_mse.axhline(base_metrics_gauss["MSE"], color="red", linestyle="--", label=f"Noisy Baseline ({base_metrics_gauss['MSE']})")
    ax_mse.set_ylabel("MSE - Lower is Better", fontsize=11)
    ax_mse.set_title("Mean Squared Error (MSE)", fontsize=12, fontweight="bold")
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(k_labels)
    ax_mse.legend(loc="upper right")
    ax_mse.grid(axis='y', linestyle=':', alpha=0.6)

    plt.suptitle("Quantitative Evaluation Metrics Comparison Across Kernel Sizes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_metrics_path = os.path.join(output_dir, "metrics_comparison_chart.png")
    plt.savefig(fig_metrics_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   * Metrics comparison chart saved: {fig_metrics_path}")

    # ------------------------------------------------------------------------
    # STEP 8: SCREENSHOT OUTPUT (AS SPECIFIED IN PDF FOLDER STRUCTURE)
    # ------------------------------------------------------------------------
    fig_dash = plt.figure(figsize=(16, 10))
    gs = fig_dash.add_gridspec(2, 3)

    ax1 = fig_dash.add_subplot(gs[0, 0])
    ax1.imshow(clean_rgb)
    ax1.set_title("(a) Clean Ground Truth", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig_dash.add_subplot(gs[0, 1])
    ax2.imshow(noisy_gaussian)
    ax2.set_title(f"(b) Gaussian Noisy (sigma=25)\nPSNR: {base_metrics_gauss['PSNR (dB)']} dB", fontsize=11)
    ax2.axis("off")

    ax3 = fig_dash.add_subplot(gs[0, 2])
    ax3.imshow(filtered_images["Gaussian Noise"]["Gauss_3"])
    ax3.set_title(f"(c) Gaussian Filter (3x3)\nPSNR: {m_g3['PSNR (dB)']} dB | SSIM: {m_g3['SSIM']}", fontsize=11, color="darkgreen", fontweight="bold")
    ax3.axis("off")

    ax4 = fig_dash.add_subplot(gs[1, 0])
    ax4.imshow(filtered_images["Gaussian Noise"]["Avg_3"])
    ax4.set_title(f"(d) Average Filter (3x3)\nPSNR: {m_a3['PSNR (dB)']} dB", fontsize=11)
    ax4.axis("off")

    # Embedded Table of Metrics
    ax_tbl = fig_dash.add_subplot(gs[1, 1:])
    ax_tbl.axis("off")
    tbl_data = df_metrics[["Noise Type", "Filter Applied", "Kernel Size", "MSE", "PSNR (dB)", "SSIM"]].values
    col_labels = ["Noise Type", "Filter Applied", "Kernel", "MSE", "PSNR (dB)", "SSIM"]
    table = ax_tbl.table(cellText=tbl_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    for k_idx, cell in table.get_celld().items():
        if k_idx[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif k_idx[0] % 2 == 1:
            cell.set_facecolor("#f2f4f4")

    fig_dash.suptitle("INNOVATIVE ASSIGNMENT - II: IMAGE NOISE REDUCTION USING CONVOLUTION\nDepartment of CSE (AI & ML) - Course Code: 21ML53T-26", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_screenshot_path = os.path.join(screenshots_dir, "output.png")
    plt.savefig(output_screenshot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Step 6] Master screenshot saved to: {output_screenshot_path}")
    print("=" * 75)
    print("Pipeline execution completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Noise Reduction Using Convolution Filters")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file (optional)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save result figures")
    parser.add_argument("--screenshots_dir", type=str, default="screenshots", help="Directory to save screenshots")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if args.image:
        target_img = os.path.abspath(args.image)
    else:
        target_img = os.path.join(project_root, "dataset", "sample_image.png")
        if not os.path.exists(target_img):
            from dataset.generate_dataset import create_synthetic_test_image
            create_synthetic_test_image(target_img)
            
    out_dir = os.path.join(project_root, args.output_dir)
    scr_dir = os.path.join(project_root, args.screenshots_dir)
    
    run_noise_reduction_experiment(
        image_path=target_img,
        output_dir=out_dir,
        screenshots_dir=scr_dir
    )
