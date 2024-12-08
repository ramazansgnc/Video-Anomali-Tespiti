import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# 3D CNN Modeli Tanımlama (Eğitimde kullanılan modelle aynı)
class CNN3DModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3DModel, self).__init__()
        self.fc = torch.nn.Linear(512, num_classes)  # Eğitime göre 512 boyutlu giriş

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten işlemi (512 boyutunda bir vektör)
        x = self.fc(x)
        return x

# Test Dataset Sınıfı
class CustomVideoDataset(Dataset):
    def __init__(self, features_dir, labels_file):
        self.features_dir = features_dir
        self.labels = self.load_labels(labels_file)
        self.files = [f for f in sorted(os.listdir(features_dir)) if f.endswith('.npy')]  # Sadece npy dosyalarını al

    def load_labels(self, labels_file):
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                file_name, label = line.strip().split(',')
                labels[file_name] = int(label)
        return labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature_file = self.files[idx]
        feature_path = os.path.join(self.features_dir, feature_file)
        features = np.load(feature_path)

        # Dosya adını ayıklayıp etiketi al
        file_name = os.path.basename(feature_file)
        genel_label = self.labels.get(file_name, 0)

        features = torch.tensor(features, dtype=torch.float32)
        genel_label = torch.tensor(genel_label, dtype=torch.long)

        return features, genel_label

# Test fonksiyonu
def test_model(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():  # Test sırasında modelin ağırlıkları güncellenmeyecek
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

# Ana Test Kısmı
if __name__ == "__main__":
    # Test etiket dosyasının yolunu belirleyin
    test_features_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/TestFeatures'
    labels_file = 'C:/Users/RAMAZAN/Desktop/VideoTespit/test_genel_etiketler.txt'
    model_path = 'trained_3dcnn_model_deneme.pth'

    # Test Dataset ve DataLoader
    test_dataset = CustomVideoDataset(features_dir=test_features_dir, labels_file=labels_file)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # Modeli yükle
    model = CNN3DModel(num_classes=2)
    model.load_state_dict(torch.load(model_path))

    # Test işlemi
    test_model(model, test_loader)