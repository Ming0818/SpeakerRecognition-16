import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

p_dir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\"
num_input = 1
num_file = 1

def processing(parentDir, filename, num):
    y, sr = librosa.load(parentDir + filename, sr=None)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=18000)

    print(S.shape)
    print(S)

    dpi = 100
    fig = plt.figure(figsize=(8.62, 0.96), dpi=dpi)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=18000, x_axis='time')
    plt.subplots_adjust(left=0, right=1, bottom=0, top = 1, wspace=0, hspace=0)
    plt.set_cmap(cmap='Greys')
    plt.axis('off')
    plt.savefig(parentDir+"melspec"+ num +".png")
    plt.show()


for i in range(num_input):
    for j in range(num_file):
        processing(p_dir + str(i+1) + "\\", "input" + str(j+1) + ".wav", str(j+1))




