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
import json
import os
import sys
import pyaudio
from rich.console import Console
from rich.markdown import Markdown
from websockets.asyncio.client import connect
from websockets.asyncio.connection import Connection
from elevenlabs import ElevenLabs, play
import numpy as np
import dotenv

dotenv.load_dotenv()

# 基础配置
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 16000
CHUNK_SIZE = 512

host = "generativelanguage.googleapis.com"
model = "gemini-2.0-flash-exp"
api_key = os.environ["GOOGLE_API_KEY"]
uri = f"wss://{host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"

# 语音设置
pya = pyaudio.PyAudio()
voice_api_key = os.environ.get("ELEVENLABS_API_KEY")
voice_model = "eleven_flash_v2_5"
voice_voice_id = "nPczCjzI2devNBz1zQrb"

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
        self.running_step = 0
        self.paused = False
        self.current_theme = None
        self.current_scenario = None
        self.console = Console()
        self.voice_client = None
        
        # 初始化语音客户端
        if voice_api_key:
            self.console.print("启动语音模式", style="green")
            self.voice_client = ElevenLabs(api_key=voice_api_key)
        else:
            self.console.print("语音模式关闭，找不到 ELEVENLABS_API_KEY", style="red")

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
                "generation_config": {"response_modalities": ["TEXT"]},
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
            try:
                if "serverContent" in response:
                    parts = response["serverContent"].get("modelTurn", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            current_response.append(part["text"])
            except Exception:
                pass

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

        self.console.print("🎤 请说英语", style="yellow")

        while True:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            data = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            if self.running_step > 1:
                continue

            # 音量检测
            audio_data = []
            for i in range(0, len(data), 2):
                sample = int.from_bytes(data[i:i+2], byteorder="little", signed=True)
                audio_data.append(abs(sample))
            volume = sum(audio_data) / len(audio_data)

            if volume > 200:
                if self.running_step == 0:
                    self.console.print("🎤 :", style="yellow", end="")
                    self.running_step += 1
                self.console.print("*", style="green", end="")
            await self.audio_out_queue.put(data)

    async def send_audio(self):
        """发送音频数据"""
        while True:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            chunk = await self.audio_out_queue.get()
            msg = {
                "realtime_input": {
                    "media_chunks": [
                        {
                            "data": base64.b64encode(chunk).decode(),
                            "mime_type": "audio/pcm",
                        }
                    ]
                }
            }
            await self.ws.send(json.dumps(msg))

    async def receive_audio(self):
        """接收和处理响应"""
        current_response = []
        async for raw_response in self.ws:
            if self.running_step == 1:
                self.console.print("\n♻️ 处理中：", end="")
                self.running_step += 1

            response = json.loads(raw_response)
            try:
                if "serverContent" in response:
                    parts = response["serverContent"].get("modelTurn", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            current_response.append(part["text"])
                            self.console.print("-", style="blue", end="")
            except Exception:
                pass

            try:
                turn_complete = response["serverContent"]["turnComplete"]
                if turn_complete and current_response:
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
                    
                    # 播放语音
                    if self.voice_client and not self.paused:
                        try:
                            def play_audio():
                                # 分割中英文内容
                                parts = text.split('---')
                                if len(parts) > 0:
                                    # 只播放英文部分（第一部分）
                                    english_text = parts[0].strip()
                                    voice_stream = self.voice_client.text_to_speech.convert_as_stream(
                                        voice_id=voice_voice_id,
                                        text=english_text,
                                        model_id=voice_model,
                                    )
                                    play(voice_stream)

                            self.console.print("🙎 声音播放中........", style="yellow")
                            await asyncio.to_thread(play_audio)
                            self.console.print("🙎 播放完毕", style="green")
                        except Exception as e:
                            self.console.print(f"语音播放错误: {e}", style="red")

                    current_response = []
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

        async with (proxy_connect(uri, proxy=proxy) if proxy else connect(uri)) as ws:
            self.ws = ws
            self.console.print("Gemini 英语口语助手", style="green", highlight=True)
            self.console.print("Make by twitter: @BoxMrChen", style="blue")
            self.console.print("============================================", style="yellow")
            
            await self.startup()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.listen_audio())
                tg.create_task(self.send_audio())
                tg.create_task(self.receive_audio())

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