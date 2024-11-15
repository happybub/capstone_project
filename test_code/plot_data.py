# -*- coding: utf-8 -*-
# @Time    : 2024/11/13 19:58
# @Author  : Gan Liyifan
# @File    : plot_data.py
from matplotlib import pyplot as plt

if __name__ == '__main__':
    jpeg = {
        'quality': [90, 80, 70, 60, 50, 40, 30, 20, 10],
        'secret_loss': [
            0.000155649,
            0.000289543,
            0.003027808,
            0.018286834,
            0.04911872,
            0.09523881,
            0.1534729,
            0.20949969,
            0.2550703
        ],
        'bit_acc': [
            1,
            0.99992675,
            0.996826172,
            0.97620237,
            0.9361023,
            0.8734497,
            0.7862854,
            0.6694458,
            0.5518249
        ]
    }

    gaussian = {
        'std': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'secret_loss': [
            0.004809819,
            0.08556767,
            0.17814203,
            0.24171333,
            0.2845852,
            0.31485224,
            0.33720607,
            0.35463077,
            0.36932865,
            0.38324398
        ],
        'bit_acc': [
            0.99992675,
            0.8983704,
            0.79907835,
            0.7340576,
            0.69071656,
            0.65855104,
            0.6342041,
            0.61502075,
            0.5987732,
            0.5851013
        ]
    }

    # Plot for JPEG
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(jpeg['quality'], jpeg['secret_loss'], label='Secret Loss', marker='o')
    plt.plot(jpeg['quality'], jpeg['bit_acc'], label='Bit Accuracy', marker='x')
    plt.xlabel('Quality')
    plt.ylabel('Value')
    plt.title('JPEG Compression')
    plt.legend()
    plt.grid(True)

    # Plot for Gaussian
    plt.subplot(1, 2, 2)
    plt.plot(gaussian['std'], gaussian['secret_loss'], label='Secret Loss', marker='o')
    plt.plot(gaussian['std'], gaussian['bit_acc'], label='Bit Accuracy', marker='x')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Value')
    plt.title('Gaussian Noise')
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.savefig('comparsion.png')
