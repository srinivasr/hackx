from typing import Callable, Any
import numpy as np

def certify_robustness(
    inference_callback: Callable[[np.ndarray], Any],
    image_bgr: np.ndarray,
    sigma: float,
    n_samples: int = 50,
    alpha: float = 0.05
) -> dict:
    """
    Implements Neyman-Pearson Randomized Smoothing (Cohen et al. 2019) to certify robustness.
    
    Args:
        inference_callback: Function that takes an image and returns a top class index or label.
        image_bgr: The clean input image (numpy array).
        sigma: Standard deviation of the Gaussian noise.
        n_samples: Number of Monte Carlo samples to draw (default 50 for speed).
        alpha: Confidence level for the certification (1 - alpha).
        
    Returns:
        dict with certification metrics, including empirical radius.
    """
    from scipy.stats import norm
    
    if sigma <= 0.0:
        # No noise means we just evaluate the clean image once.
        label = inference_callback(image_bgr)
        return {
            "certified": False,
            "top_class": label,
            "radius": 0.0,
            "p_a": 1.0,
            "message": "Sigma must be > 0 for randomized smoothing."
        }
        
    counts = {}
    
    # Monte Carlo Sampling
    for _ in range(n_samples):
        # Sample isotropic Gaussian noise
        noise = np.random.normal(0, sigma * 255.0, image_bgr.shape)
        noisy_image = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Get prediction
        prediction = inference_callback(noisy_image)
        counts[prediction] = counts.get(prediction, 0) + 1
        
    # Find top class (class A)
    if not counts:
        return {"error": "No predictions generated."}
        
    top_class = max(counts, key=counts.get)
    n_a = counts[top_class]
    
    # Calculate lower bound of the probability of class A using Clopper-Pearson
    # For a hackathon demo, we can use a simpler binomial proportion lower bound
    from scipy.stats import beta
    p_a_lower = beta.ppf(alpha, n_a, n_samples - n_a + 1) if n_a < n_samples else (1.0 - alpha)**(1.0/n_samples)
    
    # If p_a_lower > 0.5, we can certify the radius
    certified = p_a_lower > 0.5
    if certified:
        # Calculate certified radius R = sigma * Phi^-1(p_a_lower)
        radius = sigma * norm.ppf(p_a_lower)
    else:
        radius = 0.0
        
    return {
        "certified": bool(certified),
        "top_class": top_class,
        "p_a_empirical": n_a / n_samples,
        "p_a_lower_bound": float(p_a_lower),
        "certified_radius": float(radius),
        "sigma": sigma,
        "n_samples": n_samples
    }
