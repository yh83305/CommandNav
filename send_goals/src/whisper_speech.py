#!/usr/bin/env python3

import rospy
import pyaudio
import wave
from std_msgs.msg import String
import whisper
import string

filename = "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/recorded_voice.wav"


class VoiceRecorder:
    def __init__(self, filename):
        # 发布文本信息
        rospy.init_node('voice_recorder_node')
        self.speech_publisher = rospy.Publisher('/speech_topic', String, queue_size=10)
        self.filename = filename
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.frames = []
        self.p = None
        self.stream = None
        self.recording = False
        self.record_started = False
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.model = whisper.load_model("base")

    def toggle_recording(self):
        if self.record_started:
            self.stop_recording()
            self.record_started = False
        else:
            self.start_recording()
            self.record_started = True

    def start_recording(self):
        rospy.loginfo("开始录制语音...")
        self.recording = True
        self.frames = []
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self._record)
        self.stream.start_stream()

    def stop_recording(self):
        rospy.loginfo("停止录制语音...")
        self.recording = False
        if self.stream.is_active():  # 检查音频流是否处于活动状态
            self.stream.stop_stream()
        try:
            self.stream.close()
        except Exception as e:
            rospy.logwarn("关闭音频流时发生错误：%s" % str(e))
        self.p.terminate()
        self.save_to_file()
        rospy.loginfo("按下空格键开始/停止录制语音...")

        result = self.model.transcribe(self.filename)
        print("result:", result["text"])
        if result["text"] != "":
            speech = self.remove_punctuation(result["text"])
            self.speech_publisher.publish(speech)

    def _record(self, data, frame_count, time_info, status):
        self.frames.append(data)
        if not self.recording:
            return (data, pyaudio.paComplete)
        return (data, pyaudio.paContinue)

    def save_to_file(self):
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def is_english(self, text):
        return all(ord(char) < 128 for char in text)

    def remove_punctuation(self, sentence):
        # 创建一个翻译表，将标点符号映射为 None
        translator = str.maketrans('', '', string.punctuation)
        # 使用 translate() 方法移除标点符号
        return sentence.translate(translator)


def main():
    # 指定语音文件的路径
    recorder = VoiceRecorder(filename)
    rospy.loginfo("按下空格键开始/停止录制语音...")
    while not rospy.is_shutdown():
        key = input()
        if key == ' ':
            recorder.toggle_recording()
        elif key == '\x03':  # 检测到Ctrl+C按键，则退出程序
            rospy.loginfo("接收到Ctrl+C，退出程序...")
            if recorder.recording:
                recorder.stop_recording()
            rospy.signal_shutdown("User requested shutdown")


if __name__ == '__main__':
    main()
