import copy
import os
import re
import tqdm
import torch
import cv2
import io
import decord
import numpy as np
import multiprocessing as mp
from torchTraining.datasets import (
    BaseDataset,
)
from torchTraining.utils.registry import DATASETS
from torchTraining.utils.fileio import load


# Threshold for keeping a center movement per frame
c_threshold = 40
# Threshold for keeping a center vs other
d_threshold = 100000
# Values for first cropping
y1, y2, y3, y4, x1, x2, x3, x4 = (130, 230, 140, 240, 120, 240, 500, 630)  # w,h
# Final size of cropped image
width, height = (350, 240)
###################### FUNCTIONS #####################


### Function to crop image tracking the face ###
def crop_face_image(cascPATH, image, left_precenter, right_precenter, speaker):
    # Set face tracking type
    cascPATH = cascPATH
    face_cascade = cv2.CascadeClassifier(cascPATH)
    # First crop of image to simplify face-detection
    img = image.copy()
    img1 = img[y1:y2, x1:x2]
    img2 = img[y3:y4, x3:x4]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect faces (-eyes) in the image
    left_faces = face_cascade.detectMultiScale(gray1, scaleFactor=1.01, minNeighbors=5)
    right_faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.01, minNeighbors=5)
    # Track left speaker
    left_center, left_index = select_window(left_faces, left_precenter)
    # Track right speaker
    right_center, right_index = select_window(right_faces, right_precenter)
    # Calculate left crop rectangle
    left_a = left_center[0] - int(width / 2)
    left_b = left_center[1] - int(height / 2)
    # Calculate right crop rectangle
    right_a = right_center[0] - int(width / 2)
    right_b = right_center[1] - int(height / 2)

    # Select speaker to crop
    if speaker == "L":
        # Crop image
        left_a += x1
        left_b += y1
        image = image[left_b : left_b + width, left_a : left_a + height]

    elif speaker == "R":
        # Crop image
        right_a += x3
        right_b += y3
        image = image[right_b : right_b + width, right_a : right_a + height]
        # image = image[int((y3+y4)/2)-int(width/2):int((y3+y4)/2)-int(width/2)+width, int((x3+x4)/2)-int(height/2):int((x3+x4)/2)-int(height/2)+height]

    else:
        image = np.array([0, 0])

    # Return cropped image and next precenters
    return image, left_center, right_center


# Function to select the best window before cropping
def select_window(faces, precenter):
    # Case[1] of detecting no 'face'
    if np.shape(faces)[0] == 0:
        # Proposed center
        center = precenter
        # False value for index in faces of selected window
        index = -1

    else:
        # Index in faces of selected window
        index = 0
        # Case[2] of detecting many 'faces'
        if np.shape(faces)[0] > 1:
            # Starting default distance
            dmin = d_threshold
            i = 0

            # Decide which to keep
            for x, y, w, h in faces:
                # Compute center
                xc = int(round((2 * x + w) / 2))
                yc = int(round((2 * y + h) / 2))
                d = np.linalg.norm(np.array([xc, yc]) - np.array(precenter))
                # Change appropriately min and index
                if d < dmin:
                    dmin = d
                    index = i

                i += 1

            # Take values for proposed center
            index = int(index)
            x, y, w, h = faces[index]

        # Case[3] of detecting exactly one 'face'
        else:
            # Take values for proposed center
            x, y, w, h = faces[0]
            xc = int(round((2 * x + w) / 2))
            yc = int(round((2 * y + h) / 2))
            dmin = np.linalg.norm(np.array([xc, yc]) - np.array(precenter))

        # Proposed centre
        xc = int(round((2 * x + w) / 2))
        yc = int(round((2 * y + h) / 2))
        # Check distance with precenter threshold
        if dmin < c_threshold:
            # Proposed center is accepted
            center = [xc, yc]
        else:
            # Proposed center is discarded, keep precenter
            center = precenter

        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Compute big axis of face-detection rectangle
        # s1 = np.array([x, y])
        # s2 = np.array([x+w, y+h])
        # print(np.linalg.norm(s1-s2))

    if precenter == [0, 0]:
        if np.shape(faces)[0] > 0:
            x, y, w, h = faces[0]
            xc = int(round((2 * x + w) / 2))
            yc = int(round((2 * y + h) / 2))
            center = [xc, yc]
        else:
            center = [int((x2 - x1) / 2), int((y2 - y1) / 2)]

    return center, index


def capture_face_video(cascPATH, in_path, out_path, start_time, end_time, speaker):
    # First precenters don't exist
    left_precenter = [0, 0]
    right_precenter = [0, 0]

    if os.path.isfile(in_path):
        # Playing video from file
        cap = cv2.VideoCapture(in_path)
        # 读取视频帧率
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        # 设置写入视频的编码格式
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # 设置写视频的对象

        videoWriter = cv2.VideoWriter(out_path, fourcc, fps_video, (width, height))
        count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success == True:
                count += 1
                # 截取相应时间内的视频信息
                if count > (start_time * fps_video) and count <= (end_time * fps_video):
                    # 截取相应的视频帧
                    frame, left_precenter, right_precenter = crop_face_image(
                        cascPATH, frame, left_precenter, right_precenter, speaker
                    )
                    # 将图片写入视频文件
                    videoWriter.write(frame)
                if count == (end_time * fps_video):
                    break
            else:
                # 写入视频结束

                break
    # When everything is done, release the capture
    videoWriter.release()
    cap.release()
    return out_path


def capture_video(in_path, out_path, start_time, end_time, speaker):
    if os.path.isfile(in_path):
        # Playing video from file
        cap = cv2.VideoCapture(in_path)
        # 读取视频帧率
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        # 设置写入视频的编码格式
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # 设置写视频的对象

        videoWriter = cv2.VideoWriter(out_path, fourcc, fps_video, (350, 240))
        count = 0
        # 设置裁剪窗口, [宽起始点,高起始点,宽度,高度]
        window_l = [5, 120, 350, 240]
        window_r = [365, 120, 350, 240]
        while cap.isOpened():
            success, frame = cap.read()  # size: h,w
            if success == True:
                count += 1
                # 截取相应时间内的视频信息
                if count > (start_time * fps_video) and count <= (end_time * fps_video):
                    # 截取相应的视频帧

                    if speaker == "L":
                        frame = frame[
                            window_l[1] : window_l[3] + window_l[1],
                            window_l[0] : window_l[2] + window_l[0],
                            :,
                        ]
                    elif speaker == "R":
                        frame = frame[
                            window_r[1] : window_r[3] + window_r[1],
                            window_r[0] : window_r[2] + window_r[0],
                            :,
                        ]
                    # 将图片写入视频文件
                    videoWriter.write(frame)
                if count == (end_time * fps_video):
                    break
            else:
                # 写入视频结束

                break
    # When everything is done, release the capture
    videoWriter.release()
    cap.release()
    return out_path


@DATASETS.register_module()
class IEMOCAP(BaseDataset):
    def __init__(self, root, enable_video=False, threads=0, *arg, **args):
        self.enable_video = enable_video
        self.videomode == "crop"
        self.threads = threads
        if self.enable_video:
            self.check_video_file()
        super().__init__(root=root, *arg, **args)

    def load_data_list(self):
        # 文件情感标签与序号的对应关系
        tag_to_emotion = {
            "ang": "angry",
            "hap": "happy",
            "neu": "neutral",
            "sad": "sad",
            "fea": "fearful",
            "sur": "surprised",
            "dis": "disgusted",
            "exc": "excited",
            "fru": "frustrated",
            "oth": "other",
            "xxx": "xxx",
        }

        info_line = re.compile(r"\[.+\]\n", re.IGNORECASE)  # 匹配被 [] 包裹的数据行

        self._audios = []
        _labels = []
        _vad_ser = []
        self._durations = []  # durations
        _transcriptions = []  # transcriptions
        _genders = []
        _sample_rates = []

        for sess in range(1, 6):  # 遍历Session{1-5}文件夹
            # 指定和获取文件地址
            # 语音情感标签所在文件夹
            emo_evaluation_dir = os.path.join(
                self.root, f"Session{sess}/dialog/EmoEvaluation/"
            )
            evaluation_files = [
                l for l in os.listdir(emo_evaluation_dir) if (l[:3] == "Ses")
            ]  # 数据库情感结果文件列表

            # 获取文件名、情感标签、维度值、开始结束时间
            for file in evaluation_files:  # 遍历当前 Session 的 EmoEvaluation 文件
                with open(emo_evaluation_dir + file, encoding="utf-8") as f:
                    content = f.read()
                # 匹配正则化模式对应的所有内容，得到数据列表
                # 匹配被[]包裹的数据行,数据示例: [开始时间 - 结束时间] 文件名称 情感 [V, A, D]
                info_lines = re.findall(info_line, content)
                for line in info_lines[1:]:  # 忽略第一行，无用，不是想要的数据
                    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                    # .split('\t') 方法用于以制表符为基，分割此字符串
                    (
                        start_end_time,
                        wav_file_name,
                        emotion,
                        val_act_dom,
                    ) = line.strip().split("\t")
                    # 文件名，用来获取对应的语音和视频文件

                    # 获取当前语音段的开始时间和结束时间
                    start_time, end_time = start_end_time[1:-1].split("-")
                    val, act, dom = val_act_dom[1:-1].split(",")

                    self._audios.append(wav_file_name)
                    self._durations.append((float(start_time), float(end_time)))

                    # 获取情感维度值 dominance (dom), activation (act) and valence (val)
                    _vad_ser.append((float(val), float(act), float(dom)))
                    # 标签列表
                    _labels.append(tag_to_emotion[emotion])
                    # 文本翻译列表
                    _transcriptions.append("")
                    # 说话人列表
                    _genders.append(self._get_speaker(wav_file_name))
                    _sample_rates.append(16000)
            transcriptions_dir = os.path.join(
                self.root, f"Session{sess}/dialog/transcriptions/"
            )
            transcription_files = [
                l for l in os.listdir(transcriptions_dir) if (l[:3] == "Ses")
            ]
            # 语音文本翻译所在文件列表
            # 获取音频对应翻译
            for file in transcription_files:
                with open(transcriptions_dir + file, encoding="utf-8") as f:
                    for line in f.readlines():
                        try:
                            filename, _temp = line.strip().split(" [", 1)
                            text = _temp.strip().split("]: ", 1)[1]
                            # text = re.sub("\[.+\]", "", text) # 是否替换Text中的声音词，如哈哈大笑
                            index = self._audios.index(filename)
                            _transcriptions[index] = text
                        except:  # 出错就跳过
                            continue
        self.metainfo = {}
        self.data_list = {}
        self.metainfo["labels"] = sorted(list((set(_labels))))
        self.metainfo["genders"] = ["Female", "Male"]

        for i in range(len(self._audios)):
            filename_ = self._audios[i].split("_")
            self.data_list.append(
                dict(
                    audio_path=os.path.join(
                        self.root,
                        f"Session{sess}/sentences/wav/"
                        + "_".join(filename_[:-1])
                        + "/"
                        + filename
                        + ".wav",
                    ),
                    video_path=os.path.join(
                        self.root,
                        f"Session{sess}/sentences/video/"
                        + "_".join(filename_[:-1])
                        + "/"
                        + filename
                        + "_crop"
                        + ".avi",
                    )
                    if self.videomode == "crop"
                    else os.path.join(
                        self.root,
                        f"Session{sess}/sentences/video/"
                        + "_".join(filename_[:-1])
                        + "/"
                        + filename
                        + ".avi",
                    ),
                    label=_labels[i],
                    label_id=self.metainfo["labels"].index(_labels[i]),
                    vad_ser=_vad_ser[i],
                    duration=self._durations[i],
                    gender=self.metainfo["genders"].index(_genders[i]),
                    sample_rate=_sample_rates[i],
                    transcription=_transcriptions[i],
                    num_classes=len(self.metainfo["labels"]),
                )
            )
            return self.data_list

    def load_metainfo(self):
        return self.metainfo

    def check_video_file(self):
        print("检测视频文件夹是否存在...")
        fileExists = list(
            set(
                [
                    os.path.exists(
                        os.path.join(self.root, f"Session{sess}/sentences/video/")
                    )
                    for sess in range(1, 6)
                ]
            )
        )
        if len(fileExists) != 1 or fileExists[0]:
            print("检测视频文件是否损坏...")
            self.check_videoFile()
        elif not fileExists[0]:
            self._creat_video_process(self._audios)

    def check_videoFile(self) -> bool:
        # 验证处理后或已有的视频文件数量完整性...
        no_exists_filelist = []
        corrupted_filelist = []

        for filename in self._audios:
            filename_ = filename.split("_")
            sess = int(filename_[0][3:-1])
            if self.videomode == "crop":
                path = os.path.join(
                    self.root,
                    f"Session{sess}/sentences/video/"
                    + "_".join(filename_[:-1])
                    + "/"
                    + filename
                    + "_crop"
                    + ".avi",
                )
            else:
                path = os.path.join(
                    self.root,
                    f"Session{sess}/sentences/video/"
                    + "_".join(filename_[:-1])
                    + "/"
                    + filename
                    + ".avi",
                )
            if not os.path.exists(path):
                no_exists_filelist.append(filename)
            else:
                with open(path, "rb") as fh:
                    video_file = io.BytesIO(fh.read())
                try:
                    _av_reader = decord.VideoReader(
                        uri=video_file,
                        ctx=decord.cpu(0),
                        width=-1,
                        height=-1,
                        num_threads=0,
                        fault_tol=-1,
                    )
                except Exception as e:
                    corrupted_filelist.append(filename)

        if len(no_exists_filelist) > 0 or len(corrupted_filelist) > 0:
            print("丢失文件:")
            [print("\t" + i) for i in no_exists_filelist]
            print("损坏文件:")
            [print("\t" + i) for i in corrupted_filelist]
            print("尝试重新处理文件...")
            self._creat_video_process(no_exists_filelist + corrupted_filelist)
            self.check_video_file()
        else:
            print("文件完整")
        return False

    def _creat_video_process(self, filelist):
        print("自动提取脸部表情视频片段...")
        for sess in range(1, 6):
            if not os.path.exists(
                os.path.join(self.root, f"Session{sess}/sentences/video/")
            ):
                os.makedirs(os.path.join(self.root, f"Session{sess}/sentences/video/"))

        # 开始处理视频文件
        if self.threads > 0:
            length = len(filelist)
            manage = mp.Manager()
            pbar_queue = manage.Queue()
            p_pbar = mp.Process(target=self.pbar_listener, args=(pbar_queue, length))
            p_pbar.start()
            pool = []  # 进程池
            step = int(length / self.threads) + 1  # 每份的长度
            for i in range(0, length, step):
                t = mp.Process(
                    target=self._creat_video, args=(filelist[i : i + step], pbar_queue)
                )
                pool.append(t)
            for t in pool:
                t.start()
            for t in pool:
                t.join()
            pbar_queue.put(-1)  # 停止监听
            p_pbar.join()
        else:
            self._creat_video(filelist)
        print("视频文件处理成功")

    def _creat_video(self, filelist, pbar_queue=None):
        # 给定文件名列表, 处理视频文件函数
        for ind, filename in enumerate(self._audios):  # Ses01F_impro01_F000
            if filename in filelist:
                # F(L:female;R:male),M(L:male;R:female)
                mapRule = {"FF": "L", "FM": "R", "MM": "L", "MF": "R"}
                filename_ = filename.split("_")
                sess = int(filename_[0][3:-1])
                speaker = mapRule[filename_[0][-1] + filename_[-1][0]]
                # 建立当前 sess 存放视频的文件夹
                videoPath = os.path.join(
                    self.root,
                    f"Session{sess}/sentences/video/" + "_".join(filename_[:-1]),
                )
                if not os.path.exists(videoPath):
                    os.makedirs(videoPath)
                if self.videomode == "crop":
                    video_clip_path = os.path.join(
                        videoPath, filename + "_crop" + ".avi"
                    )
                else:
                    video_clip_path = os.path.join(videoPath, filename + ".avi")

                origin_video_path = os.path.join(
                    self.root,
                    f"Session{sess}/dialog/avi/DivX/"
                    + "_".join(filename_[:-1])
                    + ".avi",
                )
                if self.videomode == "crop":
                    capture_video(
                        origin_video_path,
                        video_clip_path,
                        *self._durations[ind],
                        speaker,
                    )
                else:
                    capture_face_video(
                        self.cascPATH,
                        origin_video_path,
                        video_clip_path,
                        *self._durations[ind],
                        speaker,
                    )
                if pbar_queue:
                    pbar_queue.put(1)

    def pbar_listener(self, pbar_queue, total):
        pbar = tqdm(total=total)
        pbar.set_description("处理中: ")
        while True:
            if not pbar_queue.empty():
                k = pbar_queue.get()
                if k == 1:
                    pbar.update(1)
                else:
                    break
        pbar.close()

    def _get_speaker(self, filename) -> str:
        # 给定文件名获取此文件对应说话人性别
        if "_F" in filename:
            return "Female"
        elif "_M" in filename:
            return "Male"
        else:
            raise ValueError("文件名错误")
