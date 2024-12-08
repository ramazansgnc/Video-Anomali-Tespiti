import os
import numpy as np
import cv2

# Verilerin kaydedileceği dosya yolu
save_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/islenmisTestDosyasi'

# Ardışık kareleri 16'lık segmentler halinde yükleyen fonksiyon
def load_video_from_frames(frame_files, frame_dir, num_frames=16, frame_size=(64, 64)):
    videos = []
    video_name = None

    # Her video segmenti için 16'lık gruplar oluştur
    for i in range(0, len(frame_files), num_frames):
        video_segment = []
        for j in range(num_frames):
            if i + j < len(frame_files):
                frame_file = frame_files[i + j]
                frame_path = os.path.join(frame_dir, frame_file)
                print(f"Yükleniyor: {frame_path}")
                frame = cv2.imread(frame_path)
                if frame is not None:
                    print(f"Görüntü başarıyla yüklendi: {frame_path}")
                    # Görüntüyü 64x64 boyutuna getir ve 3D CNN için boyutları ayarla
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
            video_name = frame_files[0].split('_')[0] + "_" + frame_files[0].split('_')[1]  # Dosya adından video numarasını al
            print(f'{i // num_frames + 1}. video segmenti oluşturuldu')

    return videos, video_name

# Kategoriler (Test klasöründe olanlar)
categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
              'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 
              'Shoplifting', 'Stealing', 'Vandalism']

# Ana işlem fonksiyonu (Her kategori için)
def process_and_save_videos(test_dir, save_dir):
    for category in categories:
        print(f'{category} kategorisi işleniyor...')
        category_path = os.path.join(test_dir, category)
        save_category_dir = os.path.join(save_dir, category)
        os.makedirs(save_category_dir, exist_ok=True)

        # Video türlerine göre gruplama
        video_types = {}
        
        # Kategorideki tüm dosyaları al ve video türlerine göre grupla
        for file_name in sorted(os.listdir(category_path)):
            if file_name.endswith('.png'):
                video_type = file_name.split('_')[0] + "_" + file_name.split('_')[1]  # Örneğin Abuse001, Abuse002
                if video_type not in video_types:
                    video_types[video_type] = []
                video_types[video_type].append(file_name)
        
        # Her bir video türü için işlem yap
        for video_type, frame_files in video_types.items():
            print(f'{video_type} video türü işleniyor...')

            video_segments, video_name = load_video_from_frames(frame_files, category_path, num_frames=16)

            if video_segments:  # Eğer video segmentleri boş değilse
                # Her segmenti .npy olarak kaydet
                for idx, segment in enumerate(video_segments):
                    save_path = os.path.join(save_category_dir, f'{video_name}_x264__{idx}.npy')
                    np.save(save_path, segment)
                    print(f'Segment kaydedildi: {save_path}')
            else:
                print(f'Hata: Video segmentleri boş {category_path}')
        
        print(f'{category} kategorisi tamamlandı.\n')

# Test klasörünün yolu
test_dir = r'C:/Users/RAMAZAN/Desktop/VideoTespit/Test'

# İşlem ve kaydetme fonksiyonu çağrısı
process_and_save_videos(test_dir, save_dir)
