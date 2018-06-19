import os
import subprocess
import shutil
import pytube
import librosa

#하나의 파일에 대한 다운로드와 분할
def download(link, num):
    yt = pytube.YouTube(link)  # 다운받을 동영상 URL 지정
    vids = yt.streams.all()

    # 영상 형식 리스트 확인
    # for i in range(len(vids)):
    #    print(i, '. ', vids[i])

    # 저장 경로 지정(Windows or mac)
    parent_dir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\" + num

    if os.path.exists(parent_dir):  # 이미 존재하는 경우, 삭제를 할까..
        try:
            shutil.rmtree(parent_dir)
        except OSError as e:
            if e.errno == 2:
                print  # 파일이나 디렉토리가 없음!
                'No such file or directory to remove'
                pass
            else:
                raise

    # 디렉터리 생성
    os.mkdir(parent_dir)

    vnum = 1

    # 다운로드 수행
    vids[vnum].download(parent_dir)

    # 파일 변환
    new_filename = num + '.wav'
    default_filename = vids[vnum].default_filename

    # cmd 명령어 수행
    subprocess.call(
        ['ffmpeg', '-i', os.path.join(parent_dir, default_filename), os.path.join(parent_dir, new_filename)])

    print(num + '번 동영상 다운로드 및 mp3 변환 완료!')

#유투브 링크
link = "https://www.youtube.com/watch?v=bsFZpLNwueY"

print(link)
#파일 순서(몇번째 입력인지)
num = 13

#함수 실행
download(link, str(num))

#자르기를 시작하는 위치
offset = 300

# 자르는 크기
duration = 10

pDir = "C:\\Users\\kwon\\Desktop\\개인 프로젝트\화자인식을 위한 CNN 기반 음성인식 알고리즘의 개발\\데이터\\" + str(num) + "\\"
len = librosa.get_duration(filename=pDir + str(num) + ".wav")
iteration = int((len - offset)/duration)

if iteration > 50:
    iteration = 50

for j in range(iteration):
    y, sr = librosa.load(pDir + str(num) + ".wav", sr = None, mono = True, offset = offset + j*duration, duration = duration)
    librosa.output.write_wav(pDir + "input" + str(j+1) + ".wav", y, sr)

print(str(num) + "번 파일 분할 완료!" + str(iteration) + "개 파일 생성")
