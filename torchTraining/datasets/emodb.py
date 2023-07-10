import copy
import os
import re
import tqdm
import torch

import io
import decord
import numpy as np
import soundfile as sf
import multiprocessing as mp
from torchTraining.datasets import (
    BaseDataset,
)
from torchTraining.utils.registry import DATASETS
from torchTraining.utils.fileio import load


@DATASETS.register_module()
class EMODB(BaseDataset):
    def __init__(self, root, *arg, **args):
        super().__init__(root=root, *arg, **args)

    def load_data_list(self):
        # 文件情感标签与序号的对应关系
        tag_to_emotion = {  # 文件标签与情感的对应关系
            "W": "angry",  # angry
            "L": "bored",  # boreded
            "E": "disgusted",  # disgusted
            "A": "fearful",  # fearful
            "F": "happy",  # happy
            "T": "sad",  # sad
            "N": "neutral",  # neutral
        }

        textcode = {  # 文件标签与文本的对应关系
            "a01": "Der Lappen liegt auf dem Eisschrank.",
            "a02": "Das will sie am Mittwoch abgeben.",
            "a04": "Heute abend könnte ich es ihm sagen.",
            "a05": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
            "a07": "In sieben Stunden wird es soweit sein.",
            "b01": "Was sind denn das für Tüten, die da unter dem Tisch stehen?",
            "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
            "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
            "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
            "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
        }

        male_speakers = ["03", "10", "12", "15", "11"]
        famale_speakers = ["08", "09", "13", "14", "16"]
        wavpath = os.path.join(self.root, "wav")

        self.metainfo = {}
        self.data_list = []
        self.metainfo["labels"] = sorted(list(tag_to_emotion.values()))
        self.metainfo["genders"] = ["Female", "Male"]
        # 获取指定目录下所有.wav文件列表
        files = [l for l in os.listdir(wavpath) if (".wav" in l)]
        for file in files:  # 遍历所有文件
            try:
                emotion = tag_to_emotion[os.path.basename(file)[5]]  # 获取情感标签
                text = textcode[os.path.basename(file)[2:5]]  # 获取文本
            except KeyError:
                continue
            audio_path = os.path.join(wavpath, file)  # 获取音频文件路径
            if os.path.basename(file)[0:2] in male_speakers:
                gender = "Male"
            elif os.path.basename(file)[0:2] in famale_speakers:
                gender = "Female"
            else:
                raise ValueError("文件名错误")
            gender = self.metainfo["genders"].index(gender)
            params = sf.info(os.path.join(wavpath, file))
            sample_rate = params.samplerate
            duration = params.duration
            self.data_list.append(
                dict(
                    audio_path=audio_path,
                    label=emotion,
                    label_id=self.metainfo["labels"].index(emotion),
                    duration=duration,
                    gender=gender,
                    sample_rate=sample_rate,
                    transcription=text,
                    num_classes=len(self.metainfo["labels"]),
                )
            )

        return self.data_list

    def load_metainfo(self):
        return self.metainfo
