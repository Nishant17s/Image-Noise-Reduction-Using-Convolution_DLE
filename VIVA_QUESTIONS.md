# Viva Questions & Live Modification Guide
**Course:** Deep Learning Essentials (`21ML53T-26`)  
**Topic:** Image Noise Reduction Using Convolution (`CO3` - `K3`)  

---

## 🎓 Part 1: Top Viva Questions & Answers

### Q1: What is 2D discrete convolution in image processing?
**Answer:**  
2D convolution is a spatial neighborhood operation where a small matrix called a **kernel** (or filter) slides across every pixel in an input image. At each position $(x, y)$, the element-wise product of overlapping kernel weights and image pixel intensities is calculated, and the products are summed to produce a single output pixel:
$$ (I * K)(x, y) = \sum_{m=-a}^{a} \sum_{n=-b}^{b} I(x - m, y - n) \cdot K(m, n) $$

---

### Q2: Why is the kernel flipped (by $180^\circ$) in mathematical convolution?
**Answer:**  
Mathematically, convolution includes reflection ($I(x - m, y - n)$), which requires rotating the kernel by 180 degrees (flipping horizontally and vertically). If the kernel is not flipped, the operation is called **cross-correlation**. For symmetric kernels like the Box filter and isotropic Gaussian filter, flipped kernel equals original kernel, so convolution and cross-correlation yield identical numerical outputs.

---

### Q3: Why must the sum of all elements in an Average or Gaussian kernel equal 1.0?
**Answer:**  
To **preserve the average brightness (luminance)** of the image. If $\sum K > 1.0$, the filtered image becomes progressively washed out (brighter); if $\sum K < 1.0$, the image becomes darker. Normalizing by dividing by $\sum K$ ensures zero energy drift.

---

### Q4: What is the main mathematical difference between an Average filter and a Gaussian filter?
**Answer:**  
- **Average (Box) Filter:** Assigns equal weight $1/k^2$ to all pixels in the window regardless of their distance from the center pixel.
- **Gaussian Filter:** Uses a 2D Gaussian bell-curve function $G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$. Pixels closer to the center have higher weights, while distant pixels have exponentially smaller weights. This causes much less edge blurring while smoothing noise.

---

### Q5: What is the role of parameter $\sigma$ (sigma) in a Gaussian filter?
**Answer:**  
$\sigma$ represents the standard deviation of the Gaussian distribution. It controls the spatial spread or width of the bell curve:
- A smaller $\sigma$ (e.g., $0.8 - 1.0$) concentrates weights heavily on the central pixel, providing gentle smoothing with sharp edges.
- A larger $\sigma$ (e.g., $2.0 - 3.0$) flattens the distribution across neighbor pixels, resulting in heavy smoothing and stronger blur.

---

### Q6: Why do we use reflection padding instead of zero padding?
**Answer:**  
Zero padding adds black pixels ($0$) along the borders of the image, causing dark halos and boundary distortions around image edges during convolution. Reflection padding mirrors existing border pixels, preserving edge color continuity and eliminating border artifacts.

---

### Q7: Explain the metrics MSE and PSNR.
**Answer:**  
- **MSE (Mean Squared Error):** Average squared difference between the ground-truth clean pixel and the filtered pixel:
  $$ \text{MSE} = \frac{1}{M \cdot N}\sum_{x=1}^{M}\sum_{y=1}^{N}(I_{\text{clean}}(x, y) - I_{\text{denoised}}(x, y))^2 $$
  *Lower is better.*
- **PSNR (Peak Signal-to-Noise Ratio):** Logarithmic scale measured in decibels (dB):
  $$ \text{PSNR} = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) $$
  *Higher is better.* A difference of +1 dB to +3 dB is visually noticeable.

---

### Q8: Why did Gaussian filter give a higher PSNR than Average filter?
**Answer:**  
Because Gaussian weighting treats nearby pixels with higher trust than far pixels. High-frequency noise fluctuates locally, so weighting near pixels reduces noise variance while maintaining the sharp transition steps of object boundaries, whereas the Average filter smears boundaries equally, introducing high reconstruction error (higher MSE) at every edge.

---

### Q9: Why is linear convolution less effective on Salt-and-Pepper noise?
**Answer:**  
Salt-and-Pepper noise introduces extreme outliers ($0$ or $255$). Linear filters (Average and Gaussian) perform linear weighted sums, which averages those extreme values into neighboring valid pixels, creating faded white/black spots rather than removing them. (A non-linear **Median filter** is optimal for impulse noise).

---

## 🛠️ Part 2: Live Modification Cheat Sheet (For Faculty Viva)

If the faculty asks: *"Make a live modification to verify you wrote the code"* — here are the quick recipes:

### Modification 1: "Change kernel size to 9x9"
**Where:** Open `src/image_noise_reduction.py`.  
Find line with `kernel_sizes = [3, 5, 7]`.  
Change to:
```python
kernel_sizes = [3, 5, 7, 9]
```
Re-run `python src/image_noise_reduction.py`.

---

### Modification 2: "Compute and display the Residual / Difference image"
**How:** Difference shows the noise that was removed:
```python
# Absolute difference between noisy and denoised image
residual = np.abs(clean_rgb.astype(np.float64) - gauss_filtered_g.astype(np.float64))
plt.imshow(np.clip(residual * 3, 0, 255).astype(np.uint8))
plt.title("Residual Noise Removed")
plt.show()
```

---

### Modification 3: "Add a Median Filter for Salt-and-Pepper noise"
**How:**
```python
# Apply OpenCV or SciPy median blur
median_denoised = cv2.medianBlur(noisy_sp, 3)
m_med = evaluate_image_quality(clean_rgb, median_denoised)
print(f"Median Filter PSNR: {m_med['PSNR (dB)']} dB (outperforms linear filters on impulse noise!)")
```

---

### Modification 4: "Change noise level ($\sigma=40$)"
**Where:** In `src/image_noise_reduction.py`:
```python
# Change sigma=25.0 to sigma=40.0
noisy_gaussian = add_gaussian_noise(clean_rgb, mean=0.0, sigma=40.0)
```
