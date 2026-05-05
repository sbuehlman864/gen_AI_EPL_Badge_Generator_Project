# EPL Logo-Fusion
# Main entry point for the EPL VAE logo fusion project.


import multiprocessing
from multiprocessing import reduction
import subprocess
from pathlib import Path
import json

from dotenv import load_dotenv
load_dotenv() 

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

from ray import tune
import ray
from ray import train as ray_train

import umap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_SLUG = "alexteboul/english-premier-league-logo-detection-20k-images"
DATA_DIR = Path("data")
LOGOS_DIR = DATA_DIR / "epl-logos-big" / "epl-logos-big"
OUTPUT_PATH = Path("preprocessed_imgs").resolve()
IMG_SIZE = 128


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
def preprocess_image(data_tuple) -> None:
    src_path, team_name, output_dir, img_size = data_tuple
    # We open the image and convert to RGB
    img = Image.open(src_path)
    rgb_img = img.convert("RGB")
    # We resize each image to 128 x 128 (chosen image size)
    resized_img = rgb_img.resize((img_size,img_size), Image.LANCZOS)
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
def preprocess_all_images(logos_dir: Path, output_dir: Path, num_workers: int = 4, img_size: int = 128) -> None:
    # Collect all image paths and team names
    team_image_paths = collect_image_paths(logos_dir)
    # We build our args to grab the image path, team name, output directory, and img_size
    args_for_pool = [(img, team_name, output_dir, img_size) for img, team_name in team_image_paths]
    # We use Pool to distribute the work and preprocess the images quicker
    print("Start Pool")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # We pass in our function and our args_for_pool.
        pool.map(preprocess_image, args_for_pool)
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

class Encoder(torch.nn.Module):
    def __init__(self, img_size, latent_dim) -> None:
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1) # Flatten from (batch_size,256, 8, 8) to (batch_size, 16384)
        self.mu = torch.nn.Linear(16384, self.latent_dim)
        self.log_var = torch.nn.Linear(16384, self.latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(self.latent_dim, 16384)
        self.up_sample1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.up_sample2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.up_sample3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.up_sample4 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1, 256, 8, 8) # Reshape from (batch_size, 16384) to (batch_size, 256, 8, 8)
        x = self.up_sample1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.up_sample3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.up_sample4(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.img_size, self.latent_dim)
        self.decoder = Decoder(self.img_size, self.latent_dim)
    
    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu) # epsilon as random value sampled from a normal distribution in the shape of mu
        std = torch.exp(0.5 * log_var)
        z = mu + epsilon * std
        return z
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_img = self.decoder(z)
        return (reconstructed_img, mu, log_var)

class Encoder_large(torch.nn.Module):
    def __init__(self, img_size, latent_dim) -> None:
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1) # Flatten from (batch_size,256, 8, 8) to (batch_size, 16384)
        self.mu = torch.nn.Linear(32768, self.latent_dim)
        self.log_var = torch.nn.Linear(32768, self.latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

class Decoder_large(torch.nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(self.latent_dim, 32768)
        self.up_sample1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.up_sample2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.up_sample3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.up_sample4 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1, 256, 8, 8) # Reshape from (batch_size, 16384) to (batch_size, 256, 8, 8)
        x = self.up_sample1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up_sample2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.up_sample3(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.up_sample4(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

class VAE_large(torch.nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.encoder = Encoder_large(self.img_size, self.latent_dim)
        self.decoder = Decoder_large(self.img_size, self.latent_dim)
    
    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(mu) # epsilon as random value sampled from a normal distribution in the shape of mu
        std = torch.exp(0.5 * log_var)
        z = mu + epsilon * std
        return z
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_img = self.decoder(z)
        return (reconstructed_img, mu, log_var)

def train(model, train_loader, val_loader, optimizer, epochs, scheduler, beta=1.0, checkpoint_every=5, report_to_ray=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_losses = []
    val_losses = []

    Path("model_checkpoints").mkdir(exist_ok=True)

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            images = batch[0].to(device)
            (reconstruction, mu, log_var) = model(images)
            recon_loss = torch.nn.functional.binary_cross_entropy(reconstruction, images, reduction='sum')
            kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=1)) # torch.mean makes sure scale does not change across batches
            if epoch <= 100:
              train_loss = recon_loss
            else:
              train_loss = recon_loss + beta * kl_loss
            
            train_loss.backward()
            optimizer.step()

            

            total_train_loss += train_loss.item()
        
        if checkpoint_every > 0:
            if (epoch + 1) % checkpoint_every == 0:
                    torch.save(model.state_dict(), f"model_checkpoints/model_weights_{epoch}.pt") # Save model weights every checkpoint_every epochs
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                (reconstruction, mu, log_var) = model(images)
                recon_loss = torch.nn.functional.binary_cross_entropy(reconstruction, images, reduction='sum')
                kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=1))
                if epoch <= 100:
                  val_loss = recon_loss
                else:
                  val_loss = recon_loss + beta * kl_loss
                total_val_loss += val_loss.item()
            


        train_losses.append(total_train_loss / len(train_loader)) # save average train losses per epoch
        val_losses.append(total_val_loss / len(val_loader)) # save average validation losses per epoch

        if report_to_ray:
            ray.tune.report({"val_loss": total_val_loss / len(val_loader)})

        else:
            
            if scheduler is not None:
                scheduler.step(total_val_loss / len(val_loader))

            print(f"----Epoch {epoch}----")
            print(f"Avg Train Loss: {total_train_loss / len(train_loader)}")
            print(f"Avg Val Loss: {total_val_loss / len(val_loader)}")
            print("----------------------------")

    return train_losses, val_losses




# ---------------------------------------------------------------------------
# Step 3: Hyperband Hyperparameter Tuning
# ---------------------------------------------------------------------------
def define_search_space():
    config = {
        "latent_dim": tune.choice([64, 128, 256, 512]),
        "beta": tune.uniform(1e-5, 0.1),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64])
    }
    return config


def train_trial(config, data_path, img_size, epochs):
    latent_dim = config["latent_dim"]
    beta = config["beta"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    (train_paths, val_paths) = split_train_val(data_path, 0.2)
    (train_loader, val_loader) = build_dataloaders(train_paths, val_paths, batch_size)

    # vae_model = VAE(img_size, latent_dim)
    vae_model = VAE_large(img_size, latent_dim)

    optimizer = torch.optim.Adam(vae_model.parameters(),lr)

    train_losses, val_losses = train(vae_model, train_loader, val_loader, optimizer, epochs, beta, checkpoint_every=-1,report_to_ray=True)
    return


def run_hyperband(num_samples, max_epochs, reduction_factor, data_path, img_size):
    scheduler = ray.tune.schedulers.HyperBandScheduler(
        time_attr="training_iteration",
        max_t=max_epochs,
        reduction_factor=reduction_factor,
        metric="val_loss",
        mode="min"
    )
    tuner = tune.Tuner(
        tune.with_parameters(train_trial, data_path=data_path, img_size=img_size, epochs=max_epochs),
        param_space=define_search_space(),
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler)
    )
    results = tuner.fit()
    for result in results:
        if result.error:
            print(f"Trial error: {result.error}")
    return results.get_best_result(metric="val_loss", mode="min", filter_nan_and_inf=False)


def save_best_hparams(best_config, output_path):
    best_config = best_config.config

    with open(output_path, "w") as file:
        json.dump(best_config, file)
    return

# ---------------------------------------------------------------------------
# Step 4: Latent Space Exploration
# ---------------------------------------------------------------------------

def encode_train_images(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    mus = []
    labels = []
    with torch.no_grad():
        for batch in train_loader:
            images = batch[0].to(device)
            mu, log_var = model.encoder(images)
            mus.append(mu.cpu().numpy())
            labels.append(batch[1].numpy())
        
        mus = np.concatenate(mus, axis=0)
        labels = np.concatenate(labels, axis=0)
    
    return (mus, labels)

def compute_centroids(mus, labels):
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        mask = labels == label
        centroids[label] = mus[mask].mean(axis=0) # Select all mu vectors where labels match and take the mean
    
    return centroids

def umap_visual(mus, labels):
    embedding = umap.UMAP(n_components=2).fit_transform(mus) # Reduce latent dimension down to 2D, shape of (num_images, 2)
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=5)
    plt.colorbar(label='Team')
    plt.title('UMAP Latent Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

def interpolate_centroids(centroid_a, centroid_b, alpha=0.5):
    z = (1 - alpha) * centroid_a + alpha * centroid_b
    return z

def generate_combined_img(model, z):
    z = torch.tensor(z, dtype=torch.float32) # Convert from np array to torch tensor
    z = z.unsqueeze(0) # Unsqueeze to shape (batch_size, latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)
    combined_img = (model.decoder(z)).squeeze(0).permute(1, 2, 0).cpu().detach().numpy() # Convert back to numpy
    return combined_img



# ---------------------------------------------------------------------------
# Step 5: GUI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 6: Evaluation and Presentation of Results
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
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
    
    OUTPUT_PATH.mkdir(exist_ok=True)

    preprocess_all_images(LOGOS_DIR, OUTPUT_PATH, num_workers=4, img_size=IMG_SIZE)

    # Hyperband Hyperparameters
    num_samples = 5
    max_epochs = 5
    reduction_factor = 3
    
    best_hparams = run_hyperband(num_samples, max_epochs, reduction_factor, OUTPUT_PATH, IMG_SIZE)
    save_best_hparams(best_hparams, "./best_hparams.json")

    with open("best_hparams.json", "r") as f:
        best = json.load(f)

    latent_dim = best["latent_dim"]
    beta = best["beta"]
    lr = best["lr"]
    batch_size = best["batch_size"]

    epochs = 50

    (train_paths, val_paths) = split_train_val(OUTPUT_PATH, 0.2)
    (train_loader, val_loader) = build_dataloaders(train_paths, val_paths, batch_size)

    vae_model = VAE(IMG_SIZE, latent_dim)
    

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses = train(vae_model, train_loader, val_loader, optimizer, epochs, scheduler, beta, checkpoint_every=5, report_to_ray=False)

    plot_loss_curves(train_losses, val_losses)

    mus, labels = encode_train_images(vae_model, train_loader)
    centroids = compute_centroids(mus, labels)
    umap_visual(mus, labels)

    centroid_a = centroids[0]
    centroid_b = centroids[1]
    z = interpolate_centroids(centroid_a, centroid_b, 0.5)
    combined_img = generate_combined_img(vae_model, z)
    plt.imshow(combined_img)
    plt.show()
