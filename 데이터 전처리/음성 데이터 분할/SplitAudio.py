import librosa

numOfInput = 16
offset = 300
duration = 5

for i in range(numOfInput):
    pDir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\" + str(i+1) + "\\"
    len = librosa.get_duration(filename=pDir + str(i+1) + ".wav")
    iteration = (len - offset)/duration

    if iteration > 100:
        iteration = 100

    for j in range(int(iteration)):
        y, sr = librosa.load(pDir + str(i+1) + ".wav", sr = None, mono = True, offset = offset + j*duration, duration = duration)
        librosa.output.write_wav(pDir + "input" + str(j+1) + ".wav", y, sr)

    print(str(i+1) + "번 파일 분할 완료!" + str(iteration) + "개 파일 생성")

#arr = librosa.util.frame(y, frame_length=2048, hop_length=64)
#for i in range(len(arr)) :



