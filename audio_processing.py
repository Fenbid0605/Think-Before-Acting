#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块 - 包含语音识别、音频缓冲和TTS功能
"""

import os
import time
import queue
import struct
import threading
import subprocess
import re
import numpy as np
import pyaudio
import whisper
import pyttsx3
from queue import Queue

from config import *

class AudioBuffer:
    def __init__(self, min_length=MIN_PROCESS_LENGTH):
        self.buffer = np.zeros(min_length*2, dtype=np.float32)  # 双倍长度环形缓冲
        self.min_length = min_length
        self.pointer = 0
        self._last_update = time.time()
        
    def add_chunk(self, chunk):
        # 覆盖式写入
        self._last_update = time.time()  # 记录最后更新时间
        start = self.pointer
        end = self.pointer + len(chunk)
        if end > len(self.buffer):
            overflow = end - len(self.buffer)
            self.buffer[start:] = chunk[:-overflow]
            self.buffer[:overflow] = chunk[-overflow:]
            self.pointer = overflow
        else:
            self.buffer[start:end] = chunk
            self.pointer = end % len(self.buffer)
        return self.pointer >= self.min_length  # 触发条件
    
    def need_process(self):
        return self.pointer >= self.min_length

class RealtimeTranscriber:
    def __init__(self):
        self.audio_buffer = AudioBuffer()  # 缓冲对象
        self.result_queue = Queue()  # 结果队列
        # 模型加载优化
        try:
            print(f"正在加载模型：{MODEL_PATH}")
            self.model = whisper.load_model(
                MODEL_PATH,
                device="cpu",  # 强制使用CPU
                in_memory=True   # 提升加载速度
            ).float()  # CPU必须使用float32
            self.model.eval()  # 启用评估模式减少内存占用            
            self.last_text = ""
            
            # 加载测试音频
            test_audio = whisper.load_audio("test.wav")
            trans_result = self.model.transcribe(
                test_audio,
                language=LANGUAGE,
                task="transcribe",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                no_speech_threshold=NO_SPEECH_THRESHOLD  # 增强静音过滤
            )  

            text_snippet = trans_result['text'][:20]
            print(f"测试转录结果：{text_snippet}...")

        except Exception as e:
            print(f"模型加载失败：{str(e)}")
        
    def transcribe_stream(self, is_recording_flag):
        """实时转录线程"""
        while is_recording_flag():
            try:
                # 动态触发处理
                if self.audio_buffer.need_process():
                    print(f"[{time.strftime('%H:%M:%S')}] 开始处理音频段（缓冲区填充度：{self.audio_buffer.pointer/self.audio_buffer.min_length:.1%}）")
                    # 取最新1.5秒数据（带50%重叠）
                    segment = np.concatenate([
                        self.audio_buffer.buffer[self.audio_buffer.pointer:],
                        self.audio_buffer.buffer[:self.audio_buffer.pointer]
                    ])[-self.audio_buffer.min_length:]  # 取最新min_length秒

                    result = self.model.transcribe(
                        segment,
                        language=LANGUAGE,
                        fp16=False,  # CPU必须使用FP16
                        task="transcribe",  # 明确转录任务（非翻译）
                        beam_size=BEAM_SIZE,  # 降低计算量
                        temperature=TEMPERATURE,   # 随机参数
                        no_speech_threshold=NO_SPEECH_THRESHOLD,  # 增强静音过滤
                        compression_ratio_threshold=2.2  # 抑制重复文本
                    )
                    self.result_queue.put(result["text"]) 
                    print(f"[{time.strftime('%H:%M:%S')}] 转录完成：{result['text'][:20]}...")
            except TimeoutError:
                print("转录处理超时，跳过当前片段")

class SpeakTTS:
    def __init__(self):
        # 文本内容
        self.preset_texts = [
            "好的，现在开始执行"
        ]
        
        # 初始化语音引擎
        self.engine = pyttsx3.init(driverName='espeak')
        self._configure_engine()
        
    def _configure_engine(self):
        """硬件优化配置"""
        self.engine.setProperty('rate', 160)    # 适配Nano的合理语速  160
        self.engine.setProperty('volume', 0.7)  # 防止音频削波
        self._select_chinese_voice()

    def _select_chinese_voice(self):
        """自动选择中文语音包"""
        voices = self.engine.getProperty('voices')
        for v in voices:
            if any(key in v.id.lower() for key in ['zh', 'chinese']):
                self.engine.setProperty('voice', v.id)
                print(f"已选择语音: {v.name}")
                return
        print("警告：未找到中文语音，使用默认语音")

    def sequential_play(self, interval=2):
        """开始播放文本"""
        print("开始播报...")
        for idx, text in enumerate(self.preset_texts, 1):
            print(f"正在播报第{idx}条: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            self.engine.stop()  # 重置引擎状态
            time.sleep(interval)  # 语句间隔时间
        print("所有内容播报完成")

    def say(self, text):
        """播放指定文本"""
        print(f"正在播报: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()  # 重置引擎状态

def find_xfm_device_alsa():
    """ALSA设备查找优化版"""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        xfm_match = re.search(r'card (\d+): XFMDPV0018', result.stdout)
        return int(xfm_match.group(1)) if xfm_match else None
    except Exception as e:
        print(f"ALSA设备查找失败: {e}")
        return None

def apply_volume_gain(audio_data, gain):
    """带削波保护的增益控制"""
    count = len(audio_data) // 2
    shorts = struct.unpack(f"{count}h", audio_data)
    amplified = np.clip(np.array(shorts) * gain, -32768, 32767).astype(np.int16)
    return struct.pack(f"{count}h", *amplified)

def create_audio_callback(transcriber):
    """创建音频回调函数"""
    def audio_callback(in_data, *args):
        # 直接写入缓冲区
        # 应用增益并转换格式
        amplified = apply_volume_gain(in_data, VOLUME_GAIN)
        np_data = np.frombuffer(amplified, dtype=np.int16).astype(np.float32) / 32768.0
        transcriber.audio_buffer.add_chunk(np_data)  # 直接写入类成员

        return (in_data, pyaudio.paContinue)
    
    return audio_callback 