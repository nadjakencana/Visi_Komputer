import cv2
import numpy as np
from cvzone.SegmentationModule import SelfieSegmentation

print("Mencoba memulai program...")

# 1. Inisialisasi Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")
else:
    print("Kamera ditemukan. Menginisialisasi...")

# 2. Inisialisasi Segmentasi
# model=0 (general), model=1 (landscape, lebih akurat)
segmentor = SelfieSegmentation(model=0)
print("Modul Segmentasi berhasil di-load.")

# 3. Baca gambar background
imgBG = cv2.imread("background.jpg")

if imgBG is None:
    print("Error: 'background.jpg' tidak ditemukan.")
    print("Membuat background darurat (warna hijau)...")
    # Buat background hijau darurat jika file tidak ada
    imgBG = np.full((720, 1280, 3), (0, 255, 0), dtype=np.uint8)
else:
    print("Gambar 'background.jpg' berhasil di-load.")

print("\nProgram berjalan. Tekan 'q' di jendela video untuk keluar.")

while True:
    # 4. Ambil frame dari kamera
    success, img = cap.read()
    if not success:
        print("Gagal membaca frame kamera.")
        break
    
    # 5. Samakan ukuran background dengan frame kamera
    # Ini penting agar tidak error saat digabung
    try:
        imgBG_resized = cv2.resize(imgBG, (img.shape[1], img.shape[0]))
    except Exception as e:
        print(f"Error me-resize background: {e}")
        print("Pastikan 'background.jpg' tidak rusak.")
        break

    # 6. Ini dia intinya:
    #    Pisahkan background dari orangnya
    #    img: gambar asli
    #    imgBG_resized: gambar pengganti background
    imgOut = segmentor.removeBG(img, imgBG_resized, threshold=0.1)

    # 7. Tampilkan hasil
    # Kita gabung aja jadi satu jendela biar gampang
    imgStacked = np.hstack((img, imgOut))
    cv2.imshow("Hasil (Kiri: Asli | Kanan: Background Diganti)", imgStacked)

    # 8. Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Menutup program...")
        break

# 9. Bersih-bersih
cap.release()
cv2.destroyAllWindows()
print("Program ditutup.")