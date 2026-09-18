import numpy as np
from engine.perturbation import apply_pgd_noise, generate_stress_ladder
from engine.certification import certify_robustness

def test_pgd_noise():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pgd_img = apply_pgd_noise(img, epsilon=0.05, alpha=0.01, iterations=5)
    assert pgd_img.shape == (100, 100, 3)

def test_generate_stress_ladder_ssim():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # test a random stress type
    ladder = generate_stress_ladder(img, stress_type="blur", steps=2)
    assert len(ladder) == 2
    assert "ssim_score" in ladder[0]
    
def test_certify_robustness():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # dummy inference callback that always returns 1
    def dummy_inference(img_in):
        return 1
        
    result = certify_robustness(dummy_inference, img, sigma=0.1, n_samples=10)
    assert result["top_class"] == 1
    assert "certified_radius" in result
    assert result["p_a_empirical"] == 1.0
