import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    file = r'00156.jpg'
    img = cv2.imread(file).astype(np.float32)
    bri_mean = np.mean(img)

    a = np.arange(5, 16, 5) / 10
    b = np.arange(-30, 31, 30)

    a_len = len(a)
    b_len = len(b)
    print(a_len, b_len)
    plt.figure()

    for i in range(a_len):
        for j in range(b_len):
            aa = a[i]
            bb = b[j]
            img_a = aa * (img-bri_mean) + bb + bri_mean
            print(i, j, aa, bb)
            img_a = np.clip(img_a,0,255).astype(np.uint8)
            plt.subplot(a_len+1, b_len, (j + b_len * i + 1))
            plt.imshow(img_a, cmap='gray')
    plt.subplot(a_len + 1, b_len, a_len*b_len+1)
    plt.imshow(img.astype(np.uint8), cmap='gray')
    plt.show()
