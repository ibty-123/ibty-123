
# 1. Importation des bibliothèques nécessaires
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 2. Définition du modèle U-Net pour segmentation
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = torch.cat((self.upconv4(b), e4), dim=1)
        d4 = self.decoder4(d4)
        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.output_layer(d1))

# 3. Chargement et prétraitement de l’image PET-Scan
def load_and_preprocess_image(path, target_size=(256, 256)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size) / 255.0  # Normalisation
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 4. Fonction de perte : Dice Loss
def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 5. Visualisation des résultats
def visualize_results(original, predicted):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title("Image Originale")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted.squeeze().detach().numpy(), cmap='gray')
    plt.title("Image Segmentée")
    plt.show()

# 6. Initialisation du modèle, de l'optimiseur et du DataLoader
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Chargement de l'image PET-Scan (exemple)
image = load_and_preprocess_image("path/to/pet_scan.png")
target = image.clone()  # Exemple où la cible est la même que l'image

# Préparation du DataLoader
dataset = TensorDataset(image, target)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 7. Entraînement du modèle
for epoch in range(5):  # 5 époques
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = dice_loss(output, target)
        loss.backward()
        optimizer.step()
    print(f"Époque {epoch + 1}, Perte : {loss.item():.4f}")

# 8. Prédiction et visualisation
model.eval()
with torch.no_grad():
    prediction = model(image)
visualize_results(image, prediction)
