import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 3D CNN Modeli Tanımlama
class CNN3DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3DModel, self).__init__()
        self.fc = nn.Linear(512, num_classes)  # Çıkış katmanı, sınıf sayısına göre ayarlanır

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten işlemi (512 boyutunda)
        x = self.fc(x)
        return x

# Özellik Dosyası Dataset Sınıfı
class CustomVideoDataset(Dataset):
    def __init__(self, features_dir, labels_file):
        self.features_dir = features_dir
        self.labels = self.load_labels(labels_file)
        
        # Alt klasörlerdeki tüm .npy dosyalarını topla
        self.files = []
        for root, dirs, files in os.walk(features_dir):
            for file in files:
                if file.endswith('_features.npy'):  # Sadece özellik dosyalarını al
                    self.files.append(os.path.join(root, file))  # Dosya yolunu ekle
        
        print(f"{len(self.files)} dosya bulundu. Dataset başarıyla yüklendi.")  # Debug için dosya sayısını yazdır
    
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
        features = np.load(feature_file)  # Numpy dosyasını yükle
        
        # Verilerin boyutunu kontrol et (512 boyutunda bir vektör bekliyoruz)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Dosya adını ayıklayıp etiketi al
        file_name = os.path.basename(feature_file).split('_features.npy')[0]
        label = self.labels.get(file_name, 0)  # Etiket bulunamazsa varsayılan olarak 0 ver
        
        return features, label

# Eğitim fonksiyonu
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # İleri yayılım
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Geri yayılım ve optimizasyon
            loss.backward()
            optimizer.step()
            
            # Kayıpları ve doğruluğu hesapla
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Epoch sonu çıktısı
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%')
    
    print('Eğitim tamamlandı!')

# Ana Eğitim Kısmı
if __name__ == "__main__":
    # Veriler ve etiket dosyasının yolu
    features_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/TrainFeatures'
    labels_file = 'C:/Users/RAMAZAN/Desktop/VideoTespit/genel_etiketler.txt'

    # Dataset ve DataLoader
    train_dataset = CustomVideoDataset(features_dir=features_dir, labels_file=labels_file)
    
    if len(train_dataset) == 0:
        print("Hata: Dataset boş. Dosya yollarını ve içerikleri kontrol edin.")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    # Model, Loss fonksiyonu ve optimizer tanımlama
    model = CNN3DModel(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Modelin eğitimi
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Eğitim sonrası modeli kaydet
    torch.save(model.state_dict(), 'trained_3dcnn_model_deneme.pth')
    print("Model kaydedildi.")