import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# 1. DCT: побудова матриці та блочна компресія
# -----------------------------

def build_dct_matrix(N=8):
    """
    Побудова 1D DCT-матриці розміру N×N (DCT-II).
    Використовується формула:
    D[k, n] = alpha(k) * cos( pi * (2n+1) * k / (2N) )
    """
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            if k == 0:
                alpha = np.sqrt(1.0 / N)
            else:
                alpha = np.sqrt(2.0 / N)
            D[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
    return D


def dct_compress_image(img, keep=10):
    """
    Компресія зображення за допомогою блочної DCT 8x8.
    - img: 2D масив (float64) з діапазоном [0, 255]
    - keep: кількість низькочастотних коефіцієнтів на блок (приблизно)
    Повертає:
    - rec: відновлене зображення
    - dct_coeffs: матриця DCT-коефіцієнтів (після занулення)
    """
    assert img.ndim == 2
    h, w = img.shape
    N = 8
    assert h % N == 0 and w % N == 0, "Розмір зображення має бути кратним 8"

    D = build_dct_matrix(N)

    # Матриця для збереження DCT-коефіцієнтів
    dct_coeffs = np.zeros_like(img, dtype=np.float64)

    # Маска: залишаємо k×k верхній лівий квадрат коефіцієнтів (низькі частоти)
    k_side = int(np.sqrt(keep))
    if k_side < 1:
        k_side = 1
    if k_side > N:
        k_side = N
    mask = np.zeros((N, N), dtype=bool)
    mask[:k_side, :k_side] = True

    # Пряме перетворення + "обрубання" високих частот
    for i in range(0, h, N):
        for j in range(0, w, N):
            block = img[i:i + N, j:j + N]
            B = D @ block @ D.T              # 2D-DCT через множення матриць
            Bc = np.where(mask, B, 0.0)      # обнуляємо зайві коефіцієнти
            dct_coeffs[i:i + N, j:j + N] = Bc

    # Зворотне перетворення (IDCT)
    rec = np.zeros_like(img, dtype=np.float64)
    for i in range(0, h, N):
        for j in range(0, w, N):
            Bc = dct_coeffs[i:i + N, j:j + N]
            block_rec = D.T @ Bc @ D         # IDCT: D^T * B * D
            rec[i:i + N, j:j + N] = block_rec

    rec = np.clip(rec, 0, 255)
    return rec, dct_coeffs


# -----------------------------
# 2. Haar-вейвлет: 1D, 2D, компресія
# -----------------------------

def haar_1d(signal):
    """
    Одновимірне пряме Haar-перетворення для вектора довжини N (N парне).
    Повертає [a0, a1, ..., a_{N/2-1}, d0, d1, ..., d_{N/2-1}].
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = signal.shape[0]
    assert N % 2 == 0
    a = (signal[0::2] + signal[1::2]) / np.sqrt(2.0)
    d = (signal[0::2] - signal[1::2]) / np.sqrt(2.0)
    return np.concatenate((a, d))


def inverse_haar_1d(coeffs):
    """
    Одновимірне обернене Haar-перетворення.
    На вході: [a..., d...] довжини N, N парне.
    Повертає відновлений сигнал довжини N.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    N = coeffs.shape[0]
    assert N % 2 == 0
    half = N // 2
    a = coeffs[:half]
    d = coeffs[half:]
    x_even = (a + d) / np.sqrt(2.0)
    x_odd = (a - d) / np.sqrt(2.0)
    out = np.empty(N, dtype=np.float64)
    out[0::2] = x_even
    out[1::2] = x_odd
    return out


def haar_2d(img):
    """
    Двовимірне Haar-перетворення (один рівень) для зображення.
    1) застосовується 1D Haar до кожного рядка,
    2) потім до кожного стовпця.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0

    temp = img.copy()
    # Обробка рядків
    for i in range(h):
        temp[i, :] = haar_1d(temp[i, :])

    # Обробка стовпців
    out = temp.copy()
    for j in range(w):
        out[:, j] = haar_1d(out[:, j])

    return out


def inverse_haar_2d(coeffs):
    """
    Двовимірне обернене Haar-перетворення (один рівень).
    1) спочатку інверсія по стовпцях,
    2) потім по рядках.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    h, w = coeffs.shape
    assert h % 2 == 0 and w % 2 == 0

    temp = coeffs.copy()
    # Інверсія по стовпцях
    for j in range(w):
        temp[:, j] = inverse_haar_1d(temp[:, j])

    # Інверсія по рядках
    out = temp.copy()
    for i in range(h):
        out[i, :] = inverse_haar_1d(out[i, :])

    return out


def haar_compress_image(img):
    """
    Компресія зображення за допомогою 2D Haar-вейвлета (один рівень).
    Стратегія:
    - LL (верхній лівий блок) залишаємо,
    - LH, HL, HH обнуляємо (відкидаємо деталі).
    Повертає:
    - rec: відновлене зображення
    - coeffs: початкові коефіцієнти
    - coeffs_c: коефіцієнти після компресії (з обнуленими деталями)
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0

    coeffs = haar_2d(img)
    coeffs_c = coeffs.copy()

    half_h = h // 2
    half_w = w // 2

    # Обнулення деталевих підобластей: LH, HL, HH
    coeffs_c[:half_h, half_w:] = 0.0   # LH
    coeffs_c[half_h:, :half_w] = 0.0   # HL
    coeffs_c[half_h:, half_w:] = 0.0   # HH

    rec = inverse_haar_2d(coeffs_c)
    rec = np.clip(rec, 0, 255)
    return rec, coeffs, coeffs_c


# -----------------------------
# 3. PSNR
# -----------------------------

def psnr(orig, rec, max_val=255.0):
    """
    Обчислення PSNR між оригінальним і відновленим зображеннями.
    orig, rec — 2D-масиви з однаковим розміром.
    """
    orig = np.asarray(orig, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.float64)
    mse = np.mean((orig - rec) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)


# -----------------------------
# 4. Головна функція: завантаження, компресія, візуалізація
# -----------------------------

def main():
    # ---- 4.1. Завантажити зображення ----
    # "111.png" НАЗВА ОБОВЯЗКОВА (512x512 або будь-який — ми його масштабуємо)
    image_path = "111.jpg"

    img = Image.open(image_path).convert("L")   # grayscale
    img = img.resize((512, 512))               # масштаб до 512x512
    img_np = np.array(img, dtype=np.float64)   # значення в [0,255]

    # ---- 4.2. DCT-компресія ----
    # keep = скільки низькочастотних коефіцієнтів на блок ми грубо зберігаємо
    dct_rec, dct_coeffs = dct_compress_image(img_np, keep=16)
    psnr_dct = psnr(img_np, dct_rec)

    # ---- 4.3. Haar-компресія ----
    haar_rec, coeffs, coeffs_c = haar_compress_image(img_np)
    psnr_haar = psnr(img_np, haar_rec)

    print(f"PSNR (DCT):   {psnr_dct:.2f} dB")
    print(f"PSNR (Haar):  {psnr_haar:.2f} dB")

    # ---- 4.4. Порівняльне відображення ----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(dct_rec, cmap="gray")
    plt.title(f"DCT compressed\nPSNR = {psnr_dct:.2f} dB")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(haar_rec, cmap="gray")
    plt.title(f"Haar compressed\nPSNR = {psnr_haar:.2f} dB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
