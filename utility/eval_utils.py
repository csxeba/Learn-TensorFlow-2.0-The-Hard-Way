import numpy as np


def create_montage(model, filename):
    import cv2
    samples = model.generate(n=25)
    montage = []
    for i in range(5):
        row = []
        for j in range(5):
            row.append(samples[i*5+j, ..., 0])
        montage.append(np.concatenate(row, axis=0))
    montage = np.concatenate(montage, axis=1)
    montage = (np.clip(montage, 0, 1) * 255).astype("uint8")
    cv2.imwrite(filename, montage)
