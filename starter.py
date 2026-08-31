# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from websockets.legacy.client import WebSocketClientProtocol
from websockets_proxy import Proxy, proxy_connect
import asyncio
import base64
import certifi
from collections import deque
import json
import os
import ssl
import sys
import pyaudio
from rich.console import Console
from rich.markdown import Markdown
from websockets.asyncio.client import connect
from websockets.asyncio.connection import Connection
import numpy as np
import dotenv

dotenv.load_dotenv()

# 基础配置
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512
MIN_VOICE_THRESHOLD = 500
AMBIENT_CALIBRATION_SECONDS = 1.5
SILENCE_DURATION_SECONDS = 0.8
SILENCE_CHUNKS = int(SEND_SAMPLE_RATE / CHUNK_SIZE * SILENCE_DURATION_SECONDS)
PRE_ROLL_CHUNKS = 10
PLAYBACK_TAIL_GUARD_SECONDS = 0.35
REARM_SILENCE_SECONDS = 0.6
REARM_SILENCE_CHUNKS = int(SEND_SAMPLE_RATE / CHUNK_SIZE * REARM_SILENCE_SECONDS)
VOICE_START_CONFIRM_CHUNKS = 3

host = "generativelanguage.googleapis.com"
model = "gemini-3.1-flash-live-preview"
api_key = os.environ["GOOGLE_API_KEY"]
uri = f"wss://{host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
ssl_context = ssl.create_default_context(cafile=certifi.where())

# 音频设备
pya = pyaudio.PyAudio()

# 主题和场景定义
THEMES = {
    "business": ["job interview", "business meeting", "presentation", "networking"],
    "travel": ["airport", "hotel", "restaurant", "sightseeing"],
    "daily life": ["shopping", "weather", "hobbies", "family"],
    "social": ["meeting friends", "party", "social media", "dating"],
}

class AudioLoop:
    def __init__(self):
        self.ws: WebSocketClientProtocol | Connection
        self.audio_out_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue(maxsize=64)
        self.running_step = 0
        self.speech_active = False
        self.voice_start_chunks = 0
        self.silence_chunks = 0
        self.pre_roll_audio = deque(maxlen=PRE_ROLL_CHUNKS)
        self.input_rearming = False
        self.rearm_silence_chunks = 0
        self.turn_peak_volume = 0.0
        self.paused = False
        self.current_theme = None
        self.current_scenario = None
        self.console = Console()
        self.console.print("启动 Gemini 原生语音模式", style="green")

    def calculate_pronunciation_score(self, audio_data):
        """计算发音得分"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算音频特征
            energy = np.mean(np.abs(audio_array))
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array))))
            
            # 归一化并计算得分
            energy_score = min(100, energy / 1000)
            rhythm_score = min(100, zero_crossings / 100)
            
            # 最终得分
            final_score = int(0.6 * energy_score + 0.4 * rhythm_score)
            return min(100, max(0, final_score))
        except Exception as e:
            self.console.print(f"评分计算错误: {e}", style="red")
            return 70  # 出错时返回默认分数

    async def startup(self):
        """初始化对话"""
        # 设置初始配置
        setup_msg = {
            "setup": {
                "model": f"models/{model}",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "realtimeInputConfig": {
                    "automaticActivityDetection": {"disabled": True}
                },
            }
        }
        await self.ws.send(json.dumps(setup_msg))
        await self.ws.recv()

        # 发送初始提示
        initial_msg = {
            "client_content": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": """你是一名专业的英语口语指导老师。请用中英文双语进行回复，英文在前中文在后，用 --- 分隔。
                                
Your responsibilities are:
1. Help users correct grammar and pronunciation
2. Give pronunciation scores and detailed feedback
3. Understand and respond to control commands:
   - Pause when user says "Can I have a break"
   - Continue when user says "OK let's continue"
4. Provide practice sentences based on chosen themes and scenarios

你的职责是：
1. 帮助用户纠正语法和发音
2. 给出发音评分和详细反馈
3. 理解并响应用户的控制指令：
   - 当用户说"Can I have a break"时暂停
   - 当用户说"OK let's continue"时继续
4. 基于选择的主题和场景提供练习句子

First, ask which theme they want to practice (business, travel, daily life, social) in English.

每次用户说完一个句子后，你需要：
1. 识别用户说的内容（英文）
2. 给出发音评分（0-100分）
3. 详细说明发音和语法中的问题（中英文对照）
4. 提供改进建议（中英文对照）
5. 提供下一个相关场景的练习句子（中英文对照）

请始终保持以下格式：
[English content]
---
[中文内容]

如果明白了请用中英文回答OK"""
                            }
                        ],
                    }
                ],
                "turn_complete": True,
            }
        }
        await self.ws.send(json.dumps(initial_msg))
        
        # 等待AI回复OK
        current_response = []
        async for raw_response in self.ws:
            response = json.loads(raw_response)
            text = response.get("serverContent", {}).get("outputTranscription", {}).get("text")
            if text:
                current_response.append(text)

            try:
                turn_complete = response["serverContent"]["turnComplete"]
                if turn_complete:
                    if "".join(current_response).startswith("OK"):
                        self.console.print("初始化完成 ✅", style="green")
                        return
            except KeyError:
                pass

    async def listen_audio(self):
        """监听音频输入"""
        mic_info = pya.get_default_input_device_info()
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        self.console.print("🎙️ 校准麦克风，请保持安静...", style="yellow")
        calibration_levels = []
        calibration_chunks = int(
            SEND_SAMPLE_RATE / CHUNK_SIZE * AMBIENT_CALIBRATION_SECONDS
        )
        for _ in range(calibration_chunks):
            data = await asyncio.to_thread(
                stream.read, CHUNK_SIZE, exception_on_overflow=False
            )
            samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
            if samples.size:
                calibration_levels.append(float(np.mean(np.abs(samples))))

        ambient_level = (
            float(np.percentile(calibration_levels, 95))
            if calibration_levels
            else 0
        )
        voice_start_threshold = max(MIN_VOICE_THRESHOLD, ambient_level * 2.5)
        voice_end_threshold = max(MIN_VOICE_THRESHOLD * 0.8, ambient_level * 1.5)
        self.console.print(
            f"校准完成：环境 {ambient_level:.0f}，启动阈值 {voice_start_threshold:.0f}，"
            f"结束阈值 {voice_end_threshold:.0f}",
            style="dim",
        )
        self.console.print("🎤 请说英语", style="yellow")

        while True:
            data = await asyncio.to_thread(
                stream.read, CHUNK_SIZE, exception_on_overflow=False
            )
            if self.paused or self.running_step == 2:
                continue

            samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
            volume = float(np.mean(np.abs(samples))) if samples.size else 0

            if self.input_rearming:
                if volume <= voice_end_threshold:
                    self.rearm_silence_chunks += 1
                else:
                    self.rearm_silence_chunks = 0

                if self.rearm_silence_chunks >= REARM_SILENCE_CHUNKS:
                    self.input_rearming = False
                    self.rearm_silence_chunks = 0
                    self.pre_roll_audio.clear()
                    self.console.print("🎙️ 可以继续说英语", style="yellow")
                continue

            if not self.speech_active:
                self.pre_roll_audio.append(data)
                if volume <= voice_start_threshold:
                    self.voice_start_chunks = 0
                    continue

                self.voice_start_chunks += 1
                if self.voice_start_chunks < VOICE_START_CONFIRM_CHUNKS:
                    continue

                self.speech_active = True
                self.voice_start_chunks = 0
                self.silence_chunks = 0
                self.turn_peak_volume = volume
                self.running_step = 1
                self.console.print("🎤 :", style="yellow", end="")
                await self.audio_out_queue.put(("activity_start", None))
                for buffered_chunk in self.pre_roll_audio:
                    await self.audio_out_queue.put(("audio", buffered_chunk))
                self.pre_roll_audio.clear()
                self.console.print("*", style="green", end="")
                continue

            await self.audio_out_queue.put(("audio", data))
            self.turn_peak_volume = max(self.turn_peak_volume, volume)
            if volume > voice_end_threshold:
                self.silence_chunks = 0
                self.console.print("*", style="green", end="")
                continue

            self.silence_chunks += 1
            if self.silence_chunks < SILENCE_CHUNKS:
                continue

            await self.audio_out_queue.put(("activity_end", None))
            self.speech_active = False
            self.silence_chunks = 0
            self.running_step = 2
            self.console.print(
                f"\n♻️ 处理中（麦克风峰值 {self.turn_peak_volume:.0f}）：",
                end="",
            )

    async def send_audio(self):
        """发送音频数据"""
        while True:
            message_type, chunk = await self.audio_out_queue.get()
            if message_type == "activity_start":
                msg = {"realtimeInput": {"activityStart": {}}}
            elif message_type == "activity_end":
                msg = {"realtimeInput": {"activityEnd": {}}}
            else:
                msg = {"realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(chunk).decode(),
                        "mimeType": f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                    }
                }}
            await self.ws.send(json.dumps(msg))

    async def play_audio(self):
        """流式播放 Gemini 返回的 24kHz PCM 音频。"""
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while True:
                chunk = await self.playback_queue.get()
                try:
                    await asyncio.to_thread(stream.write, chunk)
                finally:
                    self.playback_queue.task_done()
        finally:
            stream.stop_stream()
            stream.close()

    def clear_playback_queue(self):
        """丢弃尚未播放的原生音频。"""
        while True:
            try:
                self.playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self.playback_queue.task_done()

    async def receive_audio(self):
        """接收和处理响应"""
        current_response = []
        current_input_transcription = []
        playback_started = False
        async for raw_response in self.ws:
            response = json.loads(raw_response)
            server_content = response.get("serverContent", {})

            if server_content.get("interrupted"):
                self.clear_playback_queue()
                current_response = []
                current_input_transcription = []
                playback_started = False
                self.input_rearming = True
                self.rearm_silence_chunks = 0
                self.running_step = 0
                continue

            for part in server_content.get("modelTurn", {}).get("parts", []):
                inline_data = part.get("inlineData")
                if not inline_data:
                    continue
                if not playback_started:
                    self.console.print("\n🔊 Gemini 原生语音播放中...", style="yellow")
                    playback_started = True
                await self.playback_queue.put(base64.b64decode(inline_data["data"]))

            text = server_content.get("outputTranscription", {}).get("text")
            if text:
                current_response.append(text)
                self.console.print("-", style="blue", end="")

            input_text = server_content.get("inputTranscription", {}).get("text")
            if input_text:
                current_input_transcription.append(input_text)

            try:
                turn_complete = server_content["turnComplete"]
                if turn_complete:
                    if current_input_transcription:
                        self.console.print(
                            f"\n👤 Gemini 听到：{''.join(current_input_transcription).strip()}",
                            style="cyan",
                        )
                    else:
                        self.console.print("\n👤 Gemini 未返回输入转写", style="red")

                    if current_response:
                        text = "".join(current_response)
                    
                        # 检查是否是控制命令
                        if "can i have a break" in text.lower():
                            self.paused = True
                            self.console.print("\n⏸️ 会话已暂停。说 'OK let's continue' 继续", style="yellow")
                        elif "ok let's continue" in text.lower() and self.paused:
                            self.paused = False
                            self.console.print("\n▶️ 会话继续", style="green")
                    
                        # 显示响应
                        self.console.print("\n🤖 =============================================", style="yellow")
                        self.console.print(Markdown(text))

                    await self.playback_queue.join()
                    if playback_started:
                        self.console.print("🔊 Gemini 原生语音播放完毕", style="green")
                        # Keep consuming microphone frames while the speaker's
                        # acoustic tail fades, so it cannot start a new turn.
                        await asyncio.sleep(PLAYBACK_TAIL_GUARD_SECONDS)
                        self.input_rearming = True
                        self.rearm_silence_chunks = 0
                    current_response = []
                    current_input_transcription = []
                    playback_started = False
                    self.running_step = 0 if not self.paused else 2
            except KeyError:
                pass

    async def run(self):
        """主运行函数"""
        proxy = Proxy.from_url(os.environ["HTTP_PROXY"]) if os.environ.get("HTTP_PROXY") else None
        if proxy:
            self.console.print("使用代理", style="yellow")
        else:
            self.console.print("不使用代理", style="yellow")

        async with (
            proxy_connect(uri, proxy=proxy, ssl=ssl_context)
            if proxy
            else connect(uri, ssl=ssl_context)
        ) as ws:
            self.ws = ws
            self.console.print("Gemini 英语口语助手", style="green", highlight=True)
            self.console.print("Made by Twitter@BoxMrChen and Twitter@Ameowagi", style="blue")
            self.console.print("============================================", style="yellow")
            
            await self.startup()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.listen_audio())
                tg.create_task(self.send_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                def check_error(task):
                    if task.cancelled():
                        return
                    if task.exception():
                        print(f"Error: {task.exception()}")
                        sys.exit(1)

                for task in tg._tasks:
                    task.add_done_callback(check_error)

if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())
