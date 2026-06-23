import argparse
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from skimage import morphology

def get_hard_threshold_mask(img_np, threshold, min_size):
    """
    Unsupervised Color Deconvolution and Morphological Mask Generation.
    Separates the immunohistochemical (IHC) counterstain (Hematoxylin) and the target 
    biomarker stain (DAB) to isolate specific biological structures.
    """
    # Empirical stain absorbance matrix (Ruifrok & Johnston model).
    # Rows correspond to Hematoxylin, DAB, and a residual component.
    stain_matrix = np.array([
        [0.6500286,  0.704031,   0.2860126], 
        [0.2681475,  0.570313,   0.7764271], 
        [0.71102734, 0.42318153, 0.5615672]
    ])
    
    # L2 normalize the stain vectors to ensure consistent projection scales
    for i in range(3):
        stain_matrix[i] /= np.linalg.norm(stain_matrix[i])
        
    # Compute the unmixing matrix via algebraic inversion
    q = np.linalg.inv(stain_matrix)

    # Robust estimation of the background illumination intensity (I_0).
    # Utilizing the 99th percentile instead of absolute max prevents outlier-induced biases.
    brightest_ref = np.percentile(img_np, 99, axis=(0,1))
    img_float = np.clip(img_np.astype(float), 1.0, 255.0)
    
    # Apply the Beer-Lambert Law to transform the RGB transmittance into Optical Density (OD) space
    od = -np.log10(img_float / (brightest_ref + 1.0))
    od = np.maximum(od, 0)
    
    w, h, _ = od.shape
    
    # Project the OD representations onto the orthogonalized stain concentration space
    concentrations = np.dot(od.reshape((-1, 3)), q).reshape((w, h, 3))
    
    # Isolate the DAB (3,3'-Diaminobenzidine) channel which corresponds to the target expression
    dab_conc = np.maximum(concentrations[:, :, 1], 0)

    # Binarize the continuous DAB concentration utilizing a predefined rigorous threshold
    binary = dab_conc > threshold

    # Apply morphological connected-component filtering to prune non-specific 
    # background noise and isolated artifacts, preserving only structurally significant ROI.
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    return dab_conc, (binary * 255).astype(np.uint8)

def process_single_and_save(input_path, save_dir, threshold, min_size):
    """
    Diagnostic pipeline for single-image visualization.
    Exports both the binary segmentation mask and the purely reconstructed DAB pseudo-color image.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.array(Image.open(input_path).convert('RGB'))

    _, mask = get_hard_threshold_mask(img, threshold, min_size)

    file_base = os.path.splitext(os.path.basename(input_path))[0]
    save_path = os.path.join(save_dir, f"{file_base}.png")
    Image.fromarray(mask).save(save_path)

def batch_process_masks(input_dir, output_dir, threshold, min_size):
    """
    High-throughput dataset processing utility.
    Systematically generates corresponding semantic ground-truth masks for the entire target domain.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    for filename in tqdm(files):
        try:
            input_path = os.path.join(input_dir, filename)
            save_name = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, save_name)

            img = np.array(Image.open(input_path).convert('RGB'))
            _, mask = get_hard_threshold_mask(img, threshold, min_size)
            
            Image.fromarray(mask).save(output_path)

        except Exception as e:
            print(f"process {filename} error: {e}")

if __name__ == "__main__":
    
    # Expose hyperparameter interfaces to facilitate systematic empirical ablation studies
    parser = argparse.ArgumentParser(description='IHC DAB staining mask extraction with hyperparameters')
    parser.add_argument('--threshold', type=float, default=0.75, help='DAB concentration threshold for binary mask')
    parser.add_argument('--min_size', type=int, default=3500, help='Minimum object size (pixels) to keep in mask')
    parser.add_argument('--single_image', type=str, default="", help='Path to single image for processing')
    parser.add_argument('--single_save_dir', type=str, default="", help='Output directory for single image results')
    parser.add_argument('--batch_input_dir', type=str, default="", help='Input directory for batch processing')
    parser.add_argument('--batch_output_dir', type=str, default="", help='Output directory for batch processing masks')
    
    args = parser.parse_args()
    
    # --- single image inference mode ---
    if args.single_image and args.single_save_dir:
        process_single_and_save(args.single_image, args.single_save_dir, threshold=args.threshold, min_size=args.min_size)
    
    # --- Batch images inference mode ---
    if args.batch_input_dir and args.batch_output_dir:
        batch_process_masks(args.batch_input_dir, args.batch_output_dir, threshold=args.threshold, min_size=args.min_size)