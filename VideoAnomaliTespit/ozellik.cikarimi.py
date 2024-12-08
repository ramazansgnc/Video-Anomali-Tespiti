import os
import numpy as np
import torch
from torch import nn

# Model tanımlama (örneğin 3D CNN model)
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.fc = None

    def forward(self, x):
        # Conv3D ve MaxPool işlemleri
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Çıkış boyutunu dinamik olarak hesapla
        if self.fc is None:
            num_features = x.view(x.size(0), -1).size(1)
            self.fc = nn.Linear(num_features, 512).to(x.device)  # Fully connected katmanı 

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Özellik çıkarımını yapacak fonksiyon
def extract_features_from_segments(model, segment_dir, output_dir):
    model.eval()  # Modeli evaluation moduna al (eğitim modunda değil)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Her kategori için özellik çıkarımı
    for category in os.listdir(segment_dir):
        category_path = os.path.join(segment_dir, category)
        if os.path.isdir(category_path):
            output_category_dir = os.path.join(output_dir, category)
            os.makedirs(output_category_dir, exist_ok=True)
            segment_files = [f for f in os.listdir(category_path) if f.endswith('.npy')]

            for segment_file in segment_files:
                segment_path = os.path.join(category_path, segment_file)
                segment_data = np.load(segment_path, allow_pickle=True).item()
                segment = segment_data['segment']

                # Segmenti torch tensora dönüştür ve eksenleri doğru sıraya getir
                segment_tensor = torch.tensor(segment, dtype=torch.float32).to(device)
                segment_tensor = segment_tensor.permute(0, 2, 1, 3, 4)  # (1, 16, 3, 64, 64) -> (1, 3, 16, 64, 64)

                # Modelden özellikleri çıkar
                with torch.no_grad():
                    features = model(segment_tensor)

                # Özellikleri kaydet
                output_path = os.path.join(output_category_dir, f'{segment_file[:-4]}_features.npy')
                np.save(output_path, features.cpu().numpy())
                print(f'{segment_file} dosyasından özellik çıkarıldı ve {output_path} dosyasına kaydedildi.')

# Modeli yükle
model = Simple3DCNN()

# Segmentler ve çıktının kaydedileceği klasörler
segment_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/EtiketliNpy'
output_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/TrainFeatures'

# Özellik çıkarımını başlat
extract_features_from_segments(model, segment_dir, output_dir)