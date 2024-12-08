

1: Kullanılan veri setleri Train.Veri.Isleme.py  ve  Test.Veri.Isleme.py çalıştırılarak 3D CNN modeliyle özellik çıkarımı için gerekli işlemeler yapıldı.

2: Genel.Kare.etiketleme.py ve Test.GenelVeKare.Etiketleme.py çalıştırılarak islenmis verilere gerekli etiketlemeler yapılıp EtiketliNpy ve TestEtiketliNpy klasörlerinde yine segment halinde tutuldu. Yapılan etiketlemeler çıktı olarak genel_etiketler.txt ve kare_bazli_etiketler.txt şeklinde verilerek somut olarak teyit edildi.

3: ozellik.cikarimi.py ve test.ozellik.cikarimi.py çalıştırıldı ve 3D CNN modeli kullanılarak özellik çıkarımları yapıldı. Bu özellik çıkarımları TrainFeatures ve TestFeatures olarak klasörlere yine segment olarak kaydedildi.

4: egitim.py çalıştırıldı ve trained_3dcnn_model.pth isminde model eğitildi.

5: Eğitilen trained_3dcnn_model.pth Test.py çalıştırılarak test edildi.


Veri Seti(UCF Crime dataset) one drive hazır hali => https://drive.google.com/file/d/1HR3MJDDSZ4jLPd5wmafSMzjjszFdsrAg/view?usp=sharing

Kaynak(veri seti) => https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
