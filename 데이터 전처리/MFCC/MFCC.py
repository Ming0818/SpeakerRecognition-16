import librosa.display
import numpy as np
import matplotlib.pyplot as plt

p_dir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\"
num_input = 1
num_file = 1


def processing(parentDir, filename, num):
    y, sr = librosa.load(parentDir + filename, sr=None)
    librosa.feature.melspectrogram(y=y, sr=sr)

    # 고속 퓨리에 변환 결과의 절대값에 2승을 한다
    D = np.abs(librosa.stft(y)) ** 2
    S = librosa.feature.melspectrogram(S=D)


    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=20000)

    fig = plt.figure(figsize=(5, 2))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=20000, x_axis='time')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.axis('off')

    plt.savefig(parentDir + "mfcc" + num + ".png")
    #plt.show()

for i in range(num_input):
    for j in range(num_file):
        processing(p_dir + str(i + 1) + "\\", "input" + str(j + 1) + ".wav", str(j + 1))




