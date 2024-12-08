import os
import numpy as np

# Veri klasörü
segment_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/islenmisTrainDosyasi'

# Anlık etiketler dosyası
temporal_label_file = 'C:/Users/RAMAZAN/Desktop/VideoTespit/Anlık_etiketler.txt'

# Sonuçların kaydedileceği klasör ve dosyalar
output_npy_dir = 'C:/Users/RAMAZAN/Desktop/VideoTespit/EtiketliNpy'
genel_etiketler_txt = 'C:/Users/RAMAZAN/Desktop/VideoTespit/genel_etiketler.txt'
kare_bazli_etiketler_txt = 'C:/Users/RAMAZAN/Desktop/VideoTespit/kare_bazli_etiketler.txt'

os.makedirs(output_npy_dir, exist_ok=True)

# Normal ve anomali klasörlerini belirt
normal_class = 'NormalVideos'
anomaly_classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
                   'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 
                   'Stealing', 'Vandalism']

# Anlık etiketleri yükleyen fonksiyon
def load_temporal_labels(label_file):
    temporal_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Hata: '{line.strip()}' satırında eksik veri var.")
                continue
            video_name = parts[0]  # Örn: Abuse001_x264
            intervals = [(int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts), 2)]
            temporal_labels[video_name] = intervals
    return temporal_labels

# Dosyaları numaraya göre sıralayan yardımcı fonksiyon
def numeric_sort(file_list):
    sorted_list = []
    for x in file_list:
        try:
            if '_x264__' in x:
                base_name, segment = x.split('_x264__')
                segment_num = int(segment.split('.')[0])
                sorted_list.append((base_name, segment_num, x))
        except ValueError:
            print(f"Geçersiz dosya formatı: {x}, dosya atlanacak.")
    return sorted(sorted_list, key=lambda x: (x[0], x[1]))

# Kare ve genel etiketleme fonksiyonu
def label_and_save_npy(segment_dir, temporal_labels, output_npy_dir, genel_etiketler_txt, kare_bazli_etiketler_txt):
    with open(genel_etiketler_txt, 'w') as genel_f, open(kare_bazli_etiketler_txt, 'w') as kare_f:
        for category in os.listdir(segment_dir):
            category_path = os.path.join(segment_dir, category)
            if os.path.isdir(category_path):
                # Genel etiket belirle (Normal=0, Anormal=1)
                genel_label = 0 if category == normal_class else 1

                # Kategorideki dosyaları numaralı sıralamaya göre al
                segment_files = numeric_sort([sf for sf in os.listdir(category_path) if sf.endswith('.npy')])
                
                for _, segment_idx, segment_file in segment_files:
                    try:
                        # Segment dosya adını parçalarına ayır ve başlangıç karesini bul
                        video_name = segment_file.split('_x264__')[0] + '_x264'  # Örneğin Abuse001_x264
                        segment_start_frame = segment_idx * 16 * 10  # Segmentin başlangıç karesi, her kare 10x olarak ölçeklenir
                        segment_end_frame = segment_start_frame + 15 * 10  # Segmentin son karesi (10x)
                        
                        kare_bazli_labels = []  # Kare bazlı etiketleri tutmak için liste
                        segment_path = os.path.join(category_path, segment_file)
                        segment_data = np.load(segment_path)  # Segment verisini yükle

                        # Kare bazlı etiketleme
                        if video_name in temporal_labels:
                            for frame_num in range(segment_start_frame, segment_end_frame + 10, 10):
                                label = 0  # Varsayılan etiket normal
                                for start, end in temporal_labels[video_name]:
                                    if start <= frame_num <= end:
                                        label = 1  # Eğer kare anormal aralığında ise anormal olarak işaretle
                                        break
                                kare_bazli_labels.append(label)
                                frame_index_in_segment = (frame_num - segment_start_frame) // 10
                                kare_f.write(f'{segment_file}, Frame {frame_index_in_segment}, Global Frame {frame_num}, {label}\n')
                        else:
                            kare_bazli_labels = [0] * 16  # Eğer anlık etiket yoksa, tüm kareler normal kabul edilir
                            # NormalVideos klasörü içinde de tüm karelerin normal kabul edilmesi
                            if category == normal_class:
                                for frame_num in range(segment_start_frame, segment_end_frame + 10, 10):
                                    kare_f.write(f'{segment_file}, Frame {(frame_num - segment_start_frame) // 10}, Global Frame {frame_num}, 0\n')

                        # .npy dosyasına genel ve kare bazlı etiketleri kaydet
                        output_npy_path = os.path.join(output_npy_dir, category, segment_file)
                        os.makedirs(os.path.join(output_npy_dir, category), exist_ok=True)
                        np.save(output_npy_path, {'segment': segment_data, 'genel_label': genel_label, 'kare_labels': kare_bazli_labels})
                        print(f'Etiketli .npy dosyası oluşturuldu: {output_npy_path}')

                        # Genel etiketi txt dosyasına yaz
                        genel_f.write(f'{segment_file}, {genel_label}\n')

                    except IndexError:
                        print(f'Hata: {segment_file} dosyasının adı beklenen formatta değil.')

# Anlık etiketleri yükle
temporal_labels = load_temporal_labels(temporal_label_file)

# Npy dosyalarına genel ve kare bazlı etiketleri kaydet ve aynı zamanda txt dosyalarını oluştur
label_and_save_npy(segment_dir, temporal_labels, output_npy_dir, genel_etiketler_txt, kare_bazli_etiketler_txt)

print(f"İşlem tamamlandı. Sonuçlar {output_npy_dir} klasörüne kaydedildi. Ayrıca {genel_etiketler_txt} ve {kare_bazli_etiketler_txt} dosyaları oluşturuldu.")