## 说明
---
语音导航机器人，建图过程中使用open vocabulary的Detic记录图像语义，使用whisper识别语音，BERT语义理解，CLIP计算图文相似度并规划位置
## 环境安装过程
---
安装ROS-noetic
```
wget http://fishros.com/install -O fishros && . fishros
```
安装导航包
```
sudo apt-get install ros-noetic-navigation
sudo apt-get install ros-noetic-gmapping
```
创建工作空间，将仓库文件作为功能包clone到src
```
cd /home/ubuntu/Desktop/catkin_turtlebot3/src
catkin_init_workspace
git clone https://github.com/yh83305/CommandNav.git
cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```
配置soucre文件
```
sudo  gedit ~/.bashrc
```
添加
```
source /home/ubuntu/Desktop/catkin_turtlebot3/devel/setup.bash
```
然后
```
source ~/.bashrc
```
安装miniconda，然后安装环境
```
conda create -n rosp python=3.8
conda activate rosp
pip install rospkg rospy catkin_tools numpy defusedxml
pip install transformers torch sentence_transformers matplotlip
```
安装clip
```
cd /home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/CLIP-main
python setup.py build install
```
安装pyaudio
```
sudo apt-get install python-dev portaudio19-dev
pip install pyaudio
```
安装whisper
```
pip install -U openai-whisper
```
需要在https://huggingface.co/google-bert/bert-base-uncased/tree/main 中下载pytorch_model.bin，放到send_goals/src/BERT/bert-base-uncased
使用bert.py训练模型，模型命名为train_furniture2.pth
以上两个模型在send_goals/src/nlu.py中调用，可自行修改
在send_goals/src的代码中修改绝对路径
启动语义查询
```
roslaunch send_goals clip_nav.launch
```
