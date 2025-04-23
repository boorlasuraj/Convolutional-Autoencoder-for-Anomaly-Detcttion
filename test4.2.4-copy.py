import os
import re
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm  


np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on GPU: {device.type == 'cuda'}")

learning_rate = 1e-3
num_epochs = 50
batch_size = 16

data_dir_train = r"D:\conAEbackupnotebooks\brain_tumor_dataset2\Training"
data_dir_test  = r"D:\conAEbackupnotebooks\brain_tumor_dataset2\Testing"
img_size = 128


def add_gaussian_noise(images, mean=0, std=0.1):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

healthy_images = []
healthy_path = os.path.join(data_dir_train, "notumor")
for img_name in os.listdir(healthy_path):
    try:
        img_path = os.path.join(healthy_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to load image {img_name}")
            continue
        img = cv2.resize(img, (img_size, img_size))
        healthy_images.append(img)
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
healthy_images = np.array(healthy_images, dtype=np.float32) / 255.0
healthy_images = np.expand_dims(healthy_images, axis=1) 
noisy_healthy = add_gaussian_noise(healthy_images)

X_train, X_val, y_train, y_val = train_test_split(noisy_healthy, healthy_images, test_size=0.2, random_state=42)

def load_test_images(data_dir, img_size):
    test_images = []
    test_labels = [] 
    categories = ["glioma", "meningioma", "notumor", "pituitary"]
    for category in categories:
        cat_path = os.path.join(data_dir, category)
        for img_name in os.listdir(cat_path):
            try:
                img_path = os.path.join(cat_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Unable to load image {img_name} from {category}")
                    continue
                img = cv2.resize(img, (img_size, img_size))
                test_images.append(img)
                # Healthy if category is "notumor", else tumor
                test_labels.append(0 if category == "notumor" else 1)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    test_images = np.array(test_images, dtype=np.float32) / 255.0
    test_images = np.expand_dims(test_images, axis=1)
    test_labels = np.array(test_labels)
    return test_images, test_labels

all_test_images, all_test_labels = load_test_images(data_dir_test, img_size)
indices = np.random.choice(len(all_test_images), 20, replace=False)
test_images = all_test_images[indices]
test_labels = all_test_labels[indices]


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images 
        self.labels = labels  
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.squeeze(self.images[idx], axis=0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image, label

train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transform)
val_dataset = BrainTumorDataset(X_val, y_val, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)  # 64 -> 32
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(256)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(128)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64)
        )
        self.final_layer = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        dec3 = self.dec3(enc3)
        dec3_cat = torch.cat((dec3, enc2), dim=1)
        dec2 = self.dec2(dec3_cat)
        dec2_cat = torch.cat((dec2, enc1), dim=1)       
        dec1 = self.dec1(dec2_cat)
        out = self.sigmoid(self.final_layer(dec1))
        return out


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        mse = nn.functional.mse_loss(img1, img2)
        ssim_val = self.ssim(img1, img2)
        return 0.4 * mse + 0.6 * (1 - ssim_val)

    def ssim(self, img1, img2):
        mu_x = img1.mean()
        mu_y = img2.mean()
        sigma_x = img1.var()
        sigma_y = img2.var()
        sigma_xy = ((img1 - mu_x) * (img2 - mu_y)).mean()
        C1 = 0.01**2
        C2 = 0.03**2
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        return numerator / denominator


model = CAE().to(device)
criterion = SSIMLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning rate scheduler: decrease LR every 10 epochs by a factor of 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Using tqdm to show progress within each epoch
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train = running_loss / len(train_loader)
    train_losses.append(avg_train)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)
    
    # Step the learning rate scheduler
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, LR: {current_lr:.6f}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid()
plt.show()


model_save_dir = r"D:\conAEbackupnotebooks\models"  # <-- Change if needed
os.makedirs(model_save_dir, exist_ok=True)
# CHANGE: Unique filename generation for model saving
existing_model_files = [f for f in os.listdir(model_save_dir) if re.match(r"cae_model\d+\.pth", f)]
if existing_model_files:
    indices = [int(re.findall(r"\d+", f)[0]) for f in existing_model_files]
    next_index = max(indices) + 1
else:
    next_index = 1
model_save_path = os.path.join(model_save_dir, f"cae_model{next_index}.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "threshold": np.mean(np.concatenate([np.array(val_losses)]))  # placeholder threshold; will recalc during testing
}, model_save_path)
print(f"Model saved at: {model_save_path}")


model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(test_images, dtype=torch.float32).to(device)
    reconstructions = model(test_tensor).cpu().numpy()

def compute_reconstruction_error(original, reconstructed):
    return np.mean(np.abs(original - reconstructed), axis=(1,2,3))

test_errors = compute_reconstruction_error(test_images, reconstructions)

# Compute threshold from validation healthy images' reconstruction errors
val_images_list = []
for inputs, _ in val_loader:
    val_images_list.append(inputs.numpy())
val_images_concat = np.concatenate(val_images_list, axis=0)
with torch.no_grad():
    val_tensor = torch.tensor(val_images_concat, dtype=torch.float32).to(device)
    val_recon = model(val_tensor).cpu().numpy()
val_errors = compute_reconstruction_error(val_images_concat, val_recon)
threshold = np.mean(val_errors) + 2 * np.std(val_errors)
print(f"Anomaly detection threshold: {threshold:.4f}")

predictions = (test_errors > threshold).astype(int)

# Calculate no.of correct and wrong 
correct = np.sum(predictions == test_labels)
wrong = len(test_labels) - correct

class_names = ["Healthy", "Tumor"]
print("\nPredictions on Test Images:")
for i, err in enumerate(test_errors):
    status = "Tumor" if predictions[i] == 1 else "Healthy"
    true_status = "Tumor" if test_labels[i] == 1 else "Healthy"
    result = "correct" if predictions[i] == test_labels[i] else "wrong"
    print(f"Image {i+1:02d}: Recon Error = {err:.4f} -> Predicted: {status}, Actual: {true_status} ({result})")
print(f"\nTotal Correct: {correct} correct, Total Wrong: {wrong} wrong")


fig, axes = plt.subplots(20, 3, figsize=(12, 60))
for i in range(20):
    # Original Test Image
    axes[i, 0].imshow(test_images[i][0], cmap='gray')
    axes[i, 0].set_title(f"True: {class_names[test_labels[i]]}")
    axes[i, 0].axis('off')
    
    # Reconstructed Image
    axes[i, 1].imshow(reconstructions[i][0], cmap='gray')
    axes[i, 1].set_title("Reconstructed")
    axes[i, 1].axis('off')
    
    # Error Map with Prediction
    error_map = np.abs(test_images[i] - reconstructions[i])
    pred_label = class_names[predictions[i]]
    axes[i, 2].imshow(error_map[0], cmap='hot')
    axes[i, 2].set_title(f"Error: {test_errors[i]:.4f}\nPred: {pred_label}")
    axes[i, 2].axis('off')

plt.subplots_adjust(hspace=0.5, wspace=0.3)  #  spacing

output_folder = r"D:\conAEbackupnotebooks\outputs" 
os.makedirs(output_folder, exist_ok=True)
existing_output_files = [f for f in os.listdir(output_folder) if re.match(r"test_results_grid\d+\.png", f)]
if existing_output_files:
    indices = [int(re.findall(r"\d+", f)[0]) for f in existing_output_files]
    next_index = max(indices) + 1
else:
    next_index = 1
output_filename = os.path.join(output_folder, f"test_results_grid{next_index}.png")
plt.savefig(output_filename)
print(f"Test results saved to: {output_filename}")

plt.close()
print("Done.")
