# EPL Logo-Fusion
# Main entry point for the EPL VAE logo fusion project.


import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() 

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    print(f"Found {len(team_samples)} teams: {', '.join(team_samples)}")
    display_team_samples(team_samples)
