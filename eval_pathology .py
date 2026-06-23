import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.color import separate_stains, hdx_from_rgb

# Enforce academic typographic standards (e.g., IEEE/Nature formatting constraints).
# Utilizing 'Times New Roman' with 'stix' math fonts ensures publication-quality vector rendering.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['axes.unicode_minus'] = False 

def get_valid_filenames(real_folder):
    """
    Retrieves a standardized cohort of image filenames to ensure that 
    the empirical distribution comparisons are evaluated on the exact same 
    subset of histopathological tissue patches.
    """
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    filenames = [f for f in os.listdir(real_folder) if f.lower().endswith(valid_ext)]
    return filenames

def compute_folder_od_distribution(folder_path, target_filenames, bins=100, od_range=(0.1, 1.5)):
    """
    Extracts and aggregates the macroscopic Optical Density (OD) empirical distribution 
    for the target biomarker (DAB stain) across an entire dataset cohort.
    """
    total_hist = np.zeros(bins, dtype=np.float64)
    valid_count = 0
    print(f"Scanning folder: {os.path.basename(folder_path)}")
    
    for file_name in target_filenames:
        img_path = os.path.join(folder_path, file_name)
        # Fallback mechanism to resolve potential file extension mismatches (e.g., .png vs .jpg)
        # between the Source (Generated) and Target (Ground Truth) domains.
        if not os.path.exists(img_path):
            base_name = os.path.splitext(file_name)[0]
            possible_files = [f for f in os.listdir(folder_path) if f.startswith(base_name + '.')]
            if possible_files:
                img_path = os.path.join(folder_path, possible_files[0])
            else:
                continue
                
        try:
            img = io.imread(img_path)
            if img.shape[-1] == 4:
                img = color.rgba2rgb(img)
            # Map standard RGB space into Hematoxylin-DAB-Residual (HDX) optical density space
            # utilizing the built-in Ruifrok & Johnston color deconvolution formulation.
            ihc_hdx = separate_stains(img, hdx_from_rgb)
            # Index 1 corresponds exclusively to the DAB (brown) marker concentration.
            dab_od = ihc_hdx[:, :, 1]
            
            # Accumulate absolute pixel-level frequencies into the global dataset histogram
            hist, _ = np.histogram(dab_od.flatten(), bins=bins, range=od_range)
            total_hist += hist
            valid_count += 1
        except Exception as e:
            pass
            
    # Normalize the aggregated frequency histogram to formulate a valid Probability Density Function (PDF)
    if np.sum(total_hist) > 0:
        total_hist = total_hist / np.sum(total_hist)
    return total_hist

def plot_distributions(REAL_3PLUS_DIR, NETWORK_DIRS):
    """
    Constructs a comprehensive publication-ready visualization of the OD distributions.
    Overlays the generative models' predicted distributions against the true clinical 
    distribution to qualitatively and statistically assess biomarker expression fidelity.
    """
 
    # Define the quantification boundaries and resolution (bins) for the OD space.
    # OD values typically saturate around 1.5 - 2.0 in standard brightfield microscopy.
    OD_MIN = 0.1 
    OD_MAX = 1.60
    BINS = 256
    bin_edges = np.linspace(OD_MIN, OD_MAX, BINS+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
    
    target_files = get_valid_filenames(REAL_3PLUS_DIR)
    if not target_files:
        return

    # Compute the reference empirical baseline from real clinical patches
    real_dist = compute_folder_od_distribution(REAL_3PLUS_DIR, target_files, bins=BINS, od_range=(OD_MIN, OD_MAX))
    
    network_dists = {}
    for net_name, net_path in NETWORK_DIRS.items():
        dist = compute_folder_od_distribution(net_path, target_files, bins=BINS, od_range=(OD_MIN, OD_MAX))
        network_dists[net_name] = dist
        

    # Instantiate the primary figure canvas with high-resolution (300 DPI) for print quality
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300) 
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']

    # Plot the ground-truth distribution as the standard reference baseline (Black Dashed)
    ax.plot(bin_centers, real_dist, label="Real 3+ IHC", color='black', linewidth=1.2, linestyle='--')
    for i, (net_name, dist) in enumerate(network_dists.items()):
        ax.plot(bin_centers, dist, label=net_name, color=colors[i % len(colors)], linewidth=1.2, linestyle='--')

    ax.set_xlabel("Optical Density (DAB Concentration)", fontsize=14)
    ax.set_ylabel("Pixel Density", fontsize=14)

    ax.set_xlim(OD_MIN, OD_MAX)
    ax.set_ylim(-0.002, 0.15) 

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95) 

    # --- Magnified Inset Axes 1: Low-Concentration Regime ---
    # Focuses on evaluating the generative models' ability to suppress background noise
    # and accurately reconstruct weak non-specific staining.
    axins1 = ax.inset_axes([0.05, 0.52, 0.35, 0.45]) 
    
    axins1.plot(bin_centers, real_dist, color='black', linewidth=1.2, linestyle='--')
    for i, (net_name, dist) in enumerate(network_dists.items()):
        axins1.plot(bin_centers, dist, color=colors[i % len(colors)], linewidth=1.2, linestyle='--')

    axins1.set_xlim(0.1, 0.6)
    axins1.set_ylim(0.00, 0.05)  
    axins1.tick_params(axis='both', which='major', labelsize=10)
    axins1.grid(True, linestyle=':', alpha=0.4)
    ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.4)

    # --- Magnified Inset Axes 2: High-Concentration Regime ---
    # Focuses on evaluating the fine-grained distribution matching in the extreme 
    # saturation regions (crucial for distinguishing strong positive 3+ clinical cases).
    axins2 = ax.inset_axes([0.48, 0.52, 0.35, 0.45]) 
    
    axins2.plot(bin_centers, real_dist, color='black', linewidth=1.2, linestyle='--')
    for i, (net_name, dist) in enumerate(network_dists.items()):
        axins2.plot(bin_centers, dist, color=colors[i % len(colors)], linewidth=1.2, linestyle='--')

    axins2.set_xlim(1.4, 1.5)
    axins2.set_ylim(0.00, 0.015) 
    axins2.tick_params(axis='both', which='major', labelsize=10)
    axins2.grid(True, linestyle=':', alpha=0.4)
    ax.indicate_inset_zoom(axins2, edgecolor="black", alpha=0.4)

    plt.tight_layout()
    plt.savefig("OD_Distribution_Comparison.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    #Path to the folder containing real IHC images with 3+ score
    REAL_3PLUS_DIR = ""       
    
    #Path to the folder containing generated IHC images with 3+ score from each model
    NETWORK_DIRS = {
        #"CycleGAN": "",
        #"Pyramidp2p": "",
        #"ASP": "",
        #"CSSP2P": "",
        #"PSPStain": "",
        "DBMG": "",
    }

    plot_distributions(REAL_3PLUS_DIR, NETWORK_DIRS)