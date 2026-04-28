import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ATTR_COLUMNS = ["Smiling", "Male", "Eyeglasses"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vgg(device):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
    vgg = vgg.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def get_perceptual_loss(vgg, vgg_normalize, input_img, target_img):
    in_norm = vgg_normalize(input_img)
    tar_norm = vgg_normalize(target_img)
    return nn.functional.mse_loss(vgg(in_norm), vgg(tar_norm))


class CelebASharpDataset(Dataset):
    def __init__(self, image_folder, attr_path, transform):
        self.image_folder = Path(image_folder)
        self.transform = transform

        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        if not Path(attr_path).exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_path}")

        df = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        missing_columns = [column for column in ATTR_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing CelebA columns: {missing_columns}")

        self.df = df[ATTR_COLUMNS]
        self.image_names = self.df.index.to_list()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_folder / self.image_names[index]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        row = self.df.iloc[index]
        labels = torch.tensor(
            [
                1.0 if row["Smiling"] == 1 else 0.0,
                1.0 if row["Male"] == 1 else 0.0,
                1.0 if row["Eyeglasses"] == 1 else 0.0,
            ],
            dtype=torch.float32,
        )

        image = Image.open(image_path).convert("RGB")
        return self.transform(image), labels


class SharpTripleVAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim * 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        h = self.enc(x)
        return h.chunk(2, dim=-1)

    def decode_with_intervention(self, x, ds=None, dm=None, dg=None, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        z = z.clone()

        if ds is not None:
            z[:, -1] = ds
        if dm is not None:
            z[:, -2] = dm
        if dg is not None:
            z[:, -3] = dg

        return self.dec(z), mu, logvar

    def forward(self, x):
        return self.decode_with_intervention(x)


def run_balanced_training(model, loader, optimizer, vgg, vgg_normalize, device, epochs):
    print("Starting balanced training...")
    model.train()

    for epoch in range(epochs):
        for batch_index, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            recon, mu, logvar = model(imgs)

            l1_loss = nn.functional.l1_loss(recon, imgs, reduction="mean") * 200.0
            perc_loss = get_perceptual_loss(vgg, vgg_normalize, recon, imgs) * 10.0
            kld_loss = (
                -0.5
                * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                * 0.5
            )

            smile_loss = nn.functional.mse_loss(mu[:, -1], labels[:, 0]) * 500.0
            male_loss = nn.functional.mse_loss(mu[:, -2], labels[:, 1]) * 1000.0
            glasses_loss = nn.functional.mse_loss(mu[:, -3], labels[:, 2]) * 1500.0

            loss = l1_loss + perc_loss + kld_loss + smile_loss + male_loss + glasses_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_index % 300 == 0:
                print(
                    f"E{epoch + 1} B{batch_index} | "
                    f"Tot:{loss.item():.1f} | "
                    f"L1:{l1_loss.item():.1f} | "
                    f"Perc:{perc_loss.item():.1f} | "
                    f"Glasses:{glasses_loss.item():.1f}"
                )


def find_demo_target(dataset, device):
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label in test_loader:
        if label[0, 0] == 0 and label[0, 1] == 0 and label[0, 2] == 0:
            return image.to(device)

    return next(iter(test_loader))[0].to(device)


def show_results(model, dataset, device, output_path=None):
    import matplotlib.pyplot as plt

    model.eval()
    target = find_demo_target(dataset, device)

    with torch.no_grad():
        recon, _, _ = model.decode_with_intervention(target, deterministic=True)
        only_glasses = model.decode_with_intervention(
            target,
            dg=torch.tensor([1.0], device=device),
            deterministic=True,
        )[0]
        morphed = model.decode_with_intervention(
            target,
            dg=torch.tensor([1.0], device=device),
            dm=torch.tensor([1.0], device=device),
            ds=torch.tensor([1.0], device=device),
            deterministic=True,
        )[0]

    plt.figure(figsize=(15, 5))
    titles = [
        "Original (128x128)",
        "Reconstruction",
        "Do(Glasses)",
        "Do(Glasses, Male, Smile)",
    ]
    images = [target, recon, only_glasses, morphed]

    for index, image in enumerate(images):
        plt.subplot(1, 4, index + 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0))
        plt.title(titles[index])
        plt.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved result image to: {output_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CelebA causal-style VAE.")
    parser.add_argument(
        "--attr-path",
        required=True,
        help="Path to CelebA list_attr_celeba.txt.",
    )
    parser.add_argument(
        "--img-folder",
        required=True,
        help="Path to the CelebA img_align_celeba folder.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the result image instead of opening a plot window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device} | VS Code local script mode")

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    dataset = CelebASharpDataset(args.img_folder, args.attr_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vgg = build_vgg(device)
    vgg_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    model = SharpTripleVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_balanced_training(
        model=model,
        loader=loader,
        optimizer=optimizer,
        vgg=vgg,
        vgg_normalize=vgg_normalize,
        device=device,
        epochs=args.epochs,
    )
    show_results(model, dataset, device, args.output)


if __name__ == "__main__":
    main()
