# -*- coding: utf-8 -*-
# @Time    : 2024/11/13 20:14
# @Author  : Gan Liyifan
# @File    : plot_lambda.py
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lambda_comparison = {
        'l_s/l_c': ['1:99', '1:19', '1:9', '1:1', '9:1', '19:1', '99:1'],
        'secret_loss': [
            0.023,
            0.0082,
            0.00119,
            0.0004,
            0.00013,
            0.00009,
            0.00005
        ],
        'image_loss': [
            0.00068,
            0.00142,
            0.00118,
            0.0019,
            0.003,
            0.0036,
            0.00667
        ],
        'bit_acc': [
            0.97,
            0.992,
            0.9979,
            0.99985,
            0.999992,
            0.999997,
            1
        ]
    }

    # Plot secret_loss and image_loss
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_comparison['l_s/l_c'], lambda_comparison['secret_loss'], label='Secret Loss', marker='o')
    plt.plot(lambda_comparison['l_s/l_c'], lambda_comparison['image_loss'], label='Image Loss', marker='x')
    plt.xlabel('l_s / l_c')
    plt.ylabel('Value')
    plt.title('Secret Loss and Image Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('secret_image_loss.png')
    plt.show()

    # Plot bit_acc
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_comparison['l_s/l_c'], lambda_comparison['bit_acc'], label='Bit Accuracy', marker='s')
    plt.xlabel('l_s / l_c')
    plt.ylabel('Bit Accuracy')
    plt.title('Bit Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('bit_accuracy.png')
    plt.show()