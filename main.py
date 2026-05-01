# EPL Logo-Fusion
# Main entry point for the EPL VAE logo fusion project.


import multiprocessing
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() 

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_SLUG = "alexteboul/english-premier-league-logo-detection-20k-images"
DATA_DIR = Path("data")
LOGOS_DIR = DATA_DIR / "epl-logos-big" / "epl-logos-big"


# ---------------------------------------------------------------------------
# Step 1: Download dataset from Kaggle
# ---------------------------------------------------------------------------
def download_dataset():
    if LOGOS_DIR.exists():
        print(f"Dataset already present at '{LOGOS_DIR}'. Skipping download.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG,
         "--unzip", "-p", str(DATA_DIR)],
        check=True,
    )
    print("Download complete.")


# ---------------------------------------------------------------------------
# Step 1b: Collect all image paths across all team folders
# ---------------------------------------------------------------------------
def collect_image_paths(logos_dir: Path) -> list[tuple[Path, str]]:
    # Return a flat list of (image_path, team_name) tuples for every image
    team_image_paths = []
    # Loop through the logo directory in alpha order
    for team_dir in sorted(logos_dir.iterdir()):
        # Skip any loose files
        if not team_dir.is_dir():
            continue
        # Sort any images (jpg, png) inside the team folder and combine into a list
        images = sorted(team_dir.glob("*.jpg")) + sorted(team_dir.glob("*.png"))
        if images:
            # Loop over the images found for each team to pair them in a tuple with the team name and image path
            for image in images:
                team_image_paths.append((image, team_dir.name))
    # We return our collected image paths
    return team_image_paths


# ---------------------------------------------------------------------------
# Step 1c: Preprocess a single image (worker function for multiprocessing.Pool)
# ---------------------------------------------------------------------------
def preprocess_image(args: tuple[Path, str, Path]) -> None:
    src_path, team_name, output_dir = args
    # We open the image and convert to RGB
    img = Image.open(src_path)
    rgb_img = img.convert("RGB")
    # We resize each image to 128 x 128 (chosen image size)
    resized_img = rgb_img.resize((128,128), Image.LANCZOS)
    # We normalize the image and convert to a numpy array
    norm_img = np.asarray(resized_img) / 255.0
    # We save the array as a file in a subfoler with the team name
    out_dir = output_dir / team_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / src_path.name, norm_img)
    return


# ---------------------------------------------------------------------------
# Step 1d: Run parallel preprocessing with multiprocessing.Pool
# ---------------------------------------------------------------------------
def preprocess_all_images(logos_dir: Path, output_dir: Path, num_workers: int = 4) -> None:
    # Collect all image paths and team names
    team_image_paths = collect_image_paths(logos_dir)
    # We build our args to grab the image path, team name, and output directory
    args = [(img, team_name, output_dir) for img, team_name in team_image_paths]
    # We use Pool to distribute the work and preprocess the images quicker
    print("Start Pool")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # We pass in our function and our args
        pool.map(preprocess_image, args)
    print("End Pool")

    return


# ---------------------------------------------------------------------------
# Step 1e: Split preprocessed images into train and validation sets
# ---------------------------------------------------------------------------
def split_train_val(preprocessed_dir: Path, val_fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    # Gather image files from the preprocessed_dir using recursion
    images = list(preprocessed_dir.rglob("*.npy"))
    # Set our random seed and shuffle the images
    np.random.seed(42)
    np.random.shuffle(images)
    # Find the count for how many validation images we will have
    val_images = int(len(images)*val_fraction)
    # Use the val_images count to split the data
    val_paths = images[:val_images]
    train_paths = images[val_images:]
    return (train_paths, val_paths)


# ---------------------------------------------------------------------------
# Step 1f: Build PyTorch Dataset and DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(train_paths: list[Path], val_paths: list[Path], batch_size: int = 32):
    class eplDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
            # Sort team names
            unique_teams = sorted(set(p.parent.name for p in paths))
            # Map team names to index
            self.team_names = {name: i for i, name in enumerate(unique_teams)}
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, index):
            # Load in our image file, convert to a pytorch tensor
            img_file = np.load(self.paths[index])
            img = torch.tensor(img_file, dtype=torch.float32)
            # Change from HWC to CHW 
            img_chw = img.permute(2,0,1)
            # Find the team name index
            team_name = self.paths[index].parent.name
            team_idx = self.team_names[team_name]
            return (img_chw, team_idx)

    # Create train and validation datasets
    train_dataset = eplDataset(train_paths)
    val_dataset = eplDataset(val_paths)

    # Create data loaders for train and validation
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return (train_loader, val_loader)




# ---------------------------------------------------------------------------
# Step 2: VAE Architecture and Training
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step 3: Hyperband Hyperparameter Tuning
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step 4: Latent Space Exploration
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step 5: GUI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 6: Evaluation and Presentation of Results
# ---------------------------------------------------------------------------


# Code to initially view the data to understand what we are working with
# 
# ---------------------------------------------------------------------------
# Step 2: Discover team folders and pick one sample image per team
# ---------------------------------------------------------------------------
def get_one_image_per_team(logos_dir: Path) -> dict[str, Path]:
    team_samples = {}
    for team_dir in sorted(logos_dir.iterdir()):
        if not team_dir.is_dir():
            continue
        images = sorted(team_dir.glob("*.jpg")) + sorted(team_dir.glob("*.png"))
        if images:
            team_samples[team_dir.name] = images[0]
    return team_samples


# ---------------------------------------------------------------------------
# Step 3: Display one image per team in a grid
# ---------------------------------------------------------------------------
def display_team_samples(team_samples: dict[str, Path]):
    n = len(team_samples)
    cols = 5
    rows = 4

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle("EPL Logos — One Sample Per Team", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (team_name, img_path) in enumerate(team_samples.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(team_name.replace("-", " ").title(), fontsize=9)
        ax.axis("off")

    # Hide any unused subplot cells
    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).axis("off")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_dataset()

    if not LOGOS_DIR.exists():
        print(f"Expected logos folder not found: '{LOGOS_DIR}'")
        print("Contents of data/:")
        for p in sorted(DATA_DIR.rglob("*"))[:30]:
            print(" ", p)
        raise SystemExit(1)

    team_samples = get_one_image_per_team(LOGOS_DIR)
    print(team_samples)
    print(f"Found {len(team_samples)} teams: {', '.join(team_samples)}")
    display_team_samples(team_samples)
