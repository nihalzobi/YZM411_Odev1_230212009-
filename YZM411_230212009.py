""" YZM 411 Bilgisayarli Görü - ÖDEV1
    Öğrenci No : 230212009
    Ad Soyad : Nihal Zobi"""

"""
Giris dosyalari:
- normal.jpg
- golge.jpg
- lamba.jpg

Paketler:
Python 3.10.0
OpenCV 4.12.0
NumPy 2.1.3
Matplotlib 3.10.7

Parametreler:
- Parlaklik: +40
- Gama: 0.5 ve 2.0
- Kucultme orani: 1/4
- RGB/HSV esikleri: Normal goruntuden belirlenir.

Calistirma:
1. Fotograflari Python dosyasiyla ayni klasore koy.
2. Gerekli kutuphaneleri yukle.
3. python YZM411_230212009.py komutunu calistir.

Islem sirasi:
1. Piksel incelemesi
2. RGB/HSV maskeleme
3. Parlaklik ve gama donusumleri
4. Kucultme ve buyutme
"""

import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("--- Paket Sürümleri ---")
print(f"Python     : {sys.version.split()[0]}")
print(f"OpenCV     : {cv2.__version__}")
print(f"NumPy      : {np.__version__}")
print(f"Matplotlib : {matplotlib.__version__}")

# ==========================================
# HEDEF 1: GÖRÜNTÜ YAPISI VE PİKSEL ERİŞİMİ
# ==========================================
tahmin_1= (
    "Tahmin 1: Görüntülerin (H, W, 3) boyutunda, uint8 veri türünde ve 0-255 değer aralığında olmasını bekliyorum. "
    "Rubik küpün pembe, sarı ve mavi yüzeylerinden seçilen 3 pikselin RGB değerlerinin gölgede düşmesini, "
    "lamba ışığında ise sıcak aydınlatma nedeniyle özellikle R ve G kanallarının artmasını tahmin ediyorum."
)
print("\n=== HEDEF 1: Sayısal Yapı İncelemesi ===")
print(tahmin_1)

img_normal = cv2.cvtColor(cv2.imread("normal.jpg"), cv2.COLOR_BGR2RGB)
img_golge = cv2.cvtColor(cv2.imread("golge.jpg"), cv2.COLOR_BGR2RGB)
img_lamba = cv2.cvtColor(cv2.imread("lamba.jpg"), cv2.COLOR_BGR2RGB)

H, W, C = img_normal.shape
print(f"\nBoyut (H, W, C) : ({H}, {W}, {C})")
print(f"Veri Türü       : {img_normal.dtype}")
print(f"Değer Aralığı   : {img_normal.min()} - {img_normal.max()}")

pikseller = {
    "Pembe Yüzey (Üst)": (int(H * 0.365), int(W * 0.835)),
    "Sarı Yüzey (Sol Ön)": (int(H * 0.470), int(W * 0.785)),
    "Mavi Yüzey (Sağ Ön)": (int(H * 0.490), int(W * 0.855)),
}

y_orn, x_orn = pikseller["Sarı Yüzey (Sol Ön)"]
print(
    f"\nNumPy img[y, x, kanal] Erişim Gösterimi (Sarı Yüzey y={y_orn}, x={x_orn}): "
    f"R(kanal 0)={img_normal[y_orn, x_orn, 0]}, "
    f"G(kanal 1)={img_normal[y_orn, x_orn, 1]}, "
    f"B(kanal 2)={img_normal[y_orn, x_orn, 2]}"
)

print("\nSeçilen 3 Pikselin Üç Fotoğraftaki RGB Değerleri:")
for etiket, (y_p, x_p) in pikseller.items():
    print(f"  {etiket} (x={x_p}, y={y_p}):")
    print(f"    Normal : {img_normal[y_p, x_p]}")
    print(f"    Gölge  : {img_golge[y_p, x_p]}")
    print(f"    Lamba  : {img_lamba[y_p, x_p]}")


# ==========================================
# HEDEF 2: RENK EŞİKLERİ VE MASKELER
# ==========================================
tahmin_2= (
    "Tahmin 2: Aydınlatma değiştiğinde R, G ve B kanalları birlikte kayacağı için sabit RGB eşiklerinin gölgede "
    "küçülmesini, lambada ise sarımsı masa yüzeyini de hatalı seçmesini bekliyorum. HSV uzayında ton (H) kanalı "
    "parlaklıktan ayrıldığı için HSV maskesinin gölge ve lamba koşullarında nesne alanını daha kararlı koruyacağını tahmin ediyorum."
)
print("\n=== HEDEF 2: Renk Eşikleri ve Maskeler ===")
print(tahmin_2)

hsv_normal = cv2.cvtColor(img_normal, cv2.COLOR_RGB2HSV)
hsv_golge = cv2.cvtColor(img_golge, cv2.COLOR_RGB2HSV)
hsv_lamba = cv2.cvtColor(img_lamba, cv2.COLOR_RGB2HSV)

dondurulmus_esikler = {}
r = max(5, int(min(H, W) * 0.008))

for etiket, (y_p, x_p) in pikseller.items():
    roi_rgb = img_normal[y_p - r : y_p + r, x_p - r : x_p + r]
    roi_hsv = hsv_normal[y_p - r : y_p + r, x_p - r : x_p + r]

    med_rgb = np.median(roi_rgb, axis=(0, 1)).astype(int)
    med_hsv = np.median(roi_hsv, axis=(0, 1)).astype(int)

    rgb_alt = np.clip(med_rgb - 22, 0, 255).astype(np.uint8)
    rgb_ust = np.clip(med_rgb + 22, 0, 255).astype(np.uint8)

    h_m, s_m, v_m = med_hsv
    s_alt = int( max(42, s_m - 25) )
    s_ust = int( min(255, s_m + 45) )
    v_alt = int( max(80, v_m - 50) )
    v_ust = 255

    dondurulmus_esikler[etiket] = {
        "rgb_alt": rgb_alt,
        "rgb_ust": rgb_ust,
        "h_merkez": int(h_m),
        "h_tol": 8,
        "s_aralik": (s_alt, s_ust),
        "v_aralik": (v_alt, v_ust),
    }

print("\nNormal Işıktan Dondurulan Eşik Değerleri:")
for etiket, esik in dondurulmus_esikler.items():
    print(
        f"  {etiket}: RGB[{esik['rgb_alt']} - {esik['rgb_ust']}], "
        f"HSV[H={esik['h_merkez']}±{esik['h_tol']}, S={esik['s_aralik']}, V={esik['v_aralik']}]"
    )


def rgb_maske_olustur(img_rgb, esikler):
    toplam_maske = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for esik in esikler.values():
        m = cv2.inRange(img_rgb, esik["rgb_alt"], esik["rgb_ust"])
        toplam_maske = cv2.bitwise_or(toplam_maske, m)
    return toplam_maske


def hsv_maske_olustur(img_hsv, esikler):
    toplam_maske = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for esik in esikler.values():
        h_m, h_t = esik["h_merkez"], esik["h_tol"]
        s_alt, s_ust = esik["s_aralik"]
        v_alt, v_ust = esik["v_aralik"]

        if h_m - h_t < 0:
            alt1 = np.array([0, s_alt, v_alt], dtype=np.uint8)
            ust1 = np.array([h_m + h_t, s_ust, v_ust], dtype=np.uint8)
            alt2 = np.array([180 + (h_m - h_t), s_alt, v_alt], dtype=np.uint8)
            ust2 = np.array([179, s_ust, v_ust], dtype=np.uint8)
            m = cv2.bitwise_or(cv2.inRange(img_hsv, alt1, ust1), cv2.inRange(img_hsv, alt2, ust2))
        elif h_m + h_t > 179:
            alt1 = np.array([h_m - h_t, s_alt, v_alt], dtype=np.uint8)
            ust1 = np.array([179, s_ust, v_ust], dtype=np.uint8)
            alt2 = np.array([0, s_alt, v_alt], dtype=np.uint8)
            ust2 = np.array([(h_m + h_t) - 180, s_ust, v_ust], dtype=np.uint8)
            m = cv2.bitwise_or(cv2.inRange(img_hsv, alt1, ust1), cv2.inRange(img_hsv, alt2, ust2))
        else:
            alt = np.array([h_m - h_t, s_alt, v_alt], dtype=np.uint8)
            ust = np.array([h_m + h_t, s_ust, v_ust], dtype=np.uint8)
            m = cv2.inRange(img_hsv, alt, ust)

        toplam_maske = cv2.bitwise_or(toplam_maske, m)
    return toplam_maske


def alan_orani(maske):
    return (np.sum(maske > 0) / (maske.shape[0] * maske.shape[1])) * 100.0


mask_rgb_normal = rgb_maske_olustur(img_normal, dondurulmus_esikler)
mask_rgb_golge = rgb_maske_olustur(img_golge, dondurulmus_esikler)
mask_rgb_lamba = rgb_maske_olustur(img_lamba, dondurulmus_esikler)

mask_hsv_normal = hsv_maske_olustur(hsv_normal, dondurulmus_esikler)
mask_hsv_golge = hsv_maske_olustur(hsv_golge, dondurulmus_esikler)
mask_hsv_lamba = hsv_maske_olustur(hsv_lamba, dondurulmus_esikler)

print("\nMaske Alan Oranları (%):")
print(f"  RGB Normal Oran : %{alan_orani(mask_rgb_normal):.2f}")
print(f"  RGB Gölge Oran  : %{alan_orani(mask_rgb_golge):.2f}")
print(f"  RGB Lamba Oran  : %{alan_orani(mask_rgb_lamba):.2f}")
print("  " + "-" * 28)
print(f"  HSV Normal Oran : %{alan_orani(mask_hsv_normal):.2f}")
print(f"  HSV Gölge Oran  : %{alan_orani(mask_hsv_golge):.2f}")
print(f"  HSV Lamba Oran  : %{alan_orani(mask_hsv_lamba):.2f}")

fig_mask, axes = plt.subplots(3, 3, figsize=(13, 10))
basliklar_ve_gorseller = [
    ("Özgün: Normal Işık", img_normal, None),
    ("Özgün: Gölge", img_golge, None),
    ("Özgün: Lamba", img_lamba, None),
    (f"RGB Maske - Normal (%{alan_orani(mask_rgb_normal):.2f})", mask_rgb_normal, "gray"),
    (f"RGB Maske - Gölge (%{alan_orani(mask_rgb_golge):.2f})", mask_rgb_golge, "gray"),
    (f"RGB Maske - Lamba (%{alan_orani(mask_rgb_lamba):.2f})", mask_rgb_lamba, "gray"),
    (f"HSV Maske - Normal (%{alan_orani(mask_hsv_normal):.2f})", mask_hsv_normal, "gray"),
    (f"HSV Maske - Gölge (%{alan_orani(mask_hsv_golge):.2f})", mask_hsv_golge, "gray"),
    (f"HSV Maske - Lamba (%{alan_orani(mask_hsv_lamba):.2f})", mask_hsv_lamba, "gray"),
]
for ax, (baslik, gorsel, cmap_val) in zip(axes.flat, basliklar_ve_gorseller):
    ax.imshow(gorsel, cmap=cmap_val)
    ax.set_title(baslik, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("hedef2_maskeler.png", dpi=200)


# ==========================================
# HEDEF 3: PARLAKLIK VE GAMA DÖNÜŞÜMÜ
# ==========================================
tahmin_3 = (
    "Tahmin 3: b=+40 parlaklık kaydırmasında ortalamanın tam +40 artmak yerine 215 üstü piksellerin 255'te kırpılması "
    "(clipping) nedeniyle +40'tan biraz az artacağını tahmin ediyorum. Gama=0.5 dönüşümünün karanlık tonları açarak "
    "ortalamayı yükselteceğini, Gama=2.0 dönüşümünün ise görüntüyü bastırarak ortalamayı düşüreceğini bekliyorum."
)
print("\n=== HEDEF 3: Parlaklık ve Gama Dönüşümü ===")
print(tahmin_3)


def parlaklik_kaydirma(img, b=40.0):
    img_f = img.astype(np.float32)
    kirpilan = int(np.sum(img_f + b > 255.0))
    sonuc = np.clip(img_f + b, 0.0, 255.0).astype(np.uint8)
    return sonuc, kirpilan


def gama_donusumu(img, gamma):
    img_f = img.astype(np.float32)
    sonuc = np.clip(255.0 * ((img_f / 255.0) ** gamma), 0.0, 255.0).astype(np.uint8)
    return sonuc


orijinal_ort = float(np.mean(img_normal.astype(np.float32)))
parlak_img, kirpilan_sayisi = parlaklik_kaydirma(img_normal, b=40.0)
gama_05_img = gama_donusumu(img_normal, gamma=0.5)
gama_20_img = gama_donusumu(img_normal, gamma=2.0)

print(f"\nOrijinal Görüntü Ortalaması            : {orijinal_ort:.2f}")
print(f"Beklenen Teorik Ortalama (+40)         : {orijinal_ort + 40.0:.2f}")
print(f"Parlaklık +40 Sonrası Ölçülen Ortalama : {np.mean(parlak_img):.2f}")
print(f"255 Üstünde Kırpılan Kanal-Piksel Sayısı: {kirpilan_sayisi}")
print(f"Gama (y=0.5) Sonrası Ölçülen Ortalama  : {np.mean(gama_05_img):.2f}")
print(f"Gama (y=2.0) Sonrası Ölçülen Ortalama  : {np.mean(gama_20_img):.2f}")


# ==========================================
# HEDEF 4: KÜÇÜLTME VE ÖRNEKLEME
# ==========================================
tahmin_4 = (
    "Tahmin 4: Yazılı prospektüs bölgesi 1/4 oranında küçültüldüğünde yüksek frekanslı harf detaylarının kalıcı olarak "
    "kaybolacağını ve geri büyütmenin yazıyı geri getiremeyeceğini tahmin ediyorum. En yakın komşu (INTER_NEAREST) "
    "yönteminin harflerde kopukluk ve sahte desen (aliasing) üreteceğini, alan ortalamasının (INTER_AREA) ise yazıları "
    "bulanıklaştırıp grileştireceğini bekliyorum."
)
print("\n=== HEDEF 4: Örnekleme ve Küçültme ===")
print(tahmin_4)

y1, y2 = int(H * 0.70), int(H * 0.84)
x1, x2 = int(W * 0.22), int(W * 0.39)
kesit = img_normal[y1:y2, x1:x2]
h_k, w_k, _ = kesit.shape

kucuk_w, kucuk_h = w_k // 4, h_k // 4
kucuk_nearest = cv2.resize(kesit, (kucuk_w, kucuk_h), interpolation=cv2.INTER_NEAREST)
kucuk_area = cv2.resize(kesit, (kucuk_w, kucuk_h), interpolation=cv2.INTER_AREA)

buyuk_nearest = cv2.resize(kucuk_nearest, (w_k, h_k), interpolation=cv2.INTER_NEAREST)
buyuk_area = cv2.resize(kucuk_area, (w_k, h_k), interpolation=cv2.INTER_NEAREST)

print(f"\nKesit Boyutu: {w_k}x{h_k} -> 1/4 Küçültülmüş Boyut: {kucuk_w}x{kucuk_h}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Özgün Yazı Kesiti")
plt.imshow(kesit)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("1/4 INTER_NEAREST -> Büyütme (Aliasing)")
plt.imshow(buyuk_nearest)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("1/4 INTER_AREA -> Büyütme (Bulanık/Gri)")
plt.imshow(buyuk_area)
plt.axis("off")

plt.tight_layout()
plt.savefig("hedef4_ornekleme.png", dpi=200)
plt.show()