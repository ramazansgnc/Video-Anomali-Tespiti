import os
import numpy as np
import cv2  # Görüntü işlemleri için

# Verilerin kaydedileceği dosya yolu
save_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/islenmisTrainDosyasi'

# Ardışık kareleri 16'lık segmentler halinde yükleyen fonksiyon
def load_video_from_frames(frame_dir, num_frames=16, frame_size=(64, 64)):
    print(f"Görüntü dosyaları {frame_dir} klasöründen yükleniyor...")
    frame_files = sorted(os.listdir(frame_dir))  # Çerçeveler sıralı olmalı
    if len(frame_files) == 0:
        print(f"Uyarı: {frame_dir} klasöründe hiç dosya yok!")
        return []

    videos = []
    video_names = []

    # Her video segmenti için 16'lık gruplar oluşturma
    for i in range(0, len(frame_files), num_frames):
        video_segment = []
        for j in range(num_frames):
            if i + j < len(frame_files):
                frame_file = frame_files[i + j]
                frame_path = os.path.join(frame_dir, frame_file)
                print(f"Yükleniyor: {frame_path}")  # Hangi dosya yüklendiğini gör
                frame = cv2.imread(frame_path)  # Görüntüyü yükle
                if frame is not None:
                    print(f"Görüntü başarıyla yüklendi: {frame_path}")
                    # Görüntüyü 64x64 ve kanall sırası ayarlama (zaman, yükseklik, genişlik, kanal)
                    frame = cv2.resize(frame, frame_size)
                    frame = frame / 255.0  # Normalizasyon
                    frame = np.transpose(frame, (2, 0, 1))  # Kanal sırasını (Yükseklik, Genişlik, Kanal) -> (Kanal, Yükseklik, Genişlik) yap
                    video_segment.append(frame)
                else:
                    print(f"Hata: Görüntü yüklenemedi {frame_path}")
            else:
                # Eksik kare varsa, boş kare ekle (3D CNN için kanal boyutu da ekleniyor)
                empty_frame = np.zeros((3, frame_size[0], frame_size[1]))  # 3 kanal (RGB)
                video_segment.append(empty_frame)
        
        if video_segment:  # Eğer segment boş değilse
            video_segment = np.array(video_segment)  # (Zaman, Kanal, Yükseklik, Genişlik) boyutunda olmalı
            video_segment = np.expand_dims(video_segment, axis=0)  # Batch boyutu ekle (1, Zaman, Kanal, Yükseklik, Genişlik)
            videos.append(video_segment)  # (Batch, Zaman, Kanal, Yükseklik, Genişlik) boyutunda olmalı
            video_name = frame_files[i].split('_')[0]  # Dosya adından video numarasını al
            video_names.append(video_name)  # Video adı listesi (Normal_Videos001, Normal_Videos002 gibi)
            print(f'{i // num_frames + 1}. video segmenti oluşturuldu')  # Segment işlemi çıktısı
        else:
            print(f'Hata: Segment boş {frame_dir}')
    
    return videos, video_names

# Kategoriler (Train klasöründe olanlar)
categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
              'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 
              'Shoplifting', 'Stealing', 'Vandalism']

# Ana işlem fonksiyonu (Her kategori için)
def process_and_save_videos(train_dir, save_dir):
    segment_counter = 0  # Tüm videolar arasında kesintisiz segment numarası oluşturmak için
    for category in categories:
        print(f'{category} kategorisi işleniyor...')  # Kategori çıktısı
        category_path = os.path.join(train_dir, category)
        save_category_dir = os.path.join(save_dir, category)
        os.makedirs(save_category_dir, exist_ok=True)  # Kaydedilecek klasörü oluştur
        
        # Her kategori içindeki görüntü dosyalarını işleyelim
        video_segments, video_names = load_video_from_frames(category_path, num_frames=16)
        
        if video_segments:  # Eğer video segmentleri boş değilse
            # Her segmenti .npy olarak kaydet
            for idx, segment in enumerate(video_segments):
                video_name = video_names[idx]  # Örn: Normal_Videos001
                save_path = os.path.join(save_category_dir, f'{video_name}_x264__{segment_counter}.npy')
                np.save(save_path, segment)
                print(f'Segment kaydedildi: {save_path}')  # Segment kaydedildi çıktısı
                segment_counter += 1  # Kesintisiz segment numarası için sayaç arttır
        else:
            print(f'Hata: Video segmentleri boş {category_path}')
        
        print(f'{category} kategorisi tamamlandı.\n')  # Kategori tamamlandığında çıktı

# Train klasörünün yolu
train_dir = r'C:/Users/RAMAZAN/Desktop/VideoTespit/Train'

# İşlem ve kaydetme fonksiyonu çağrısı
process_and_save_videos(train_dir, save_dir)
