import os
import librosa

numOfInput = 12
iteration = 100
Dir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\"

for i in range(numOfInput):
    pDir = Dir + str(i+1) + "\\"

    for j in range(iteration):
        filename = pDir + "input" + str(j+1) + ".wav"
        if os.path.exists(filename): #해당 파일이 없는 것은 x표시된 파일이 있는 거임
            os.remove(filename)
