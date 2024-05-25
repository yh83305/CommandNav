安装ROS-noetic<br />
wget http://fishros.com/install -O fishros && . fishros<br />
安装导航包<br />
sudo apt-get install ros-noetic-navigation<br />
sudo apt-get install ros-noetic-gmapping<br />
创建工作空间，将仓库文件作为功能包clone到src<br />
cd /home/ubuntu/Desktop/catkin_turtlebot3/src<br />
catkin_init_workspace<br />
git clone https://github.com/yh83305/CommandNav.git<br />
cd ..<br />
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3<br />
配置soucre文件<br />
sudo  gedit ~/.bashrc<br />
添加"source /home/ubuntu/Desktop/catkin_turtlebot3/devel/setup.bash"<br />
source ~/.bashrc<br />
安装miniconda，然后安装环境<br />
conda create -n rosp python=3.8<br />
conda activate rosp<br />
pip install rospkg rospy catkin_tools numpy defusedxml<br />
pip install transformers torch sentence_transformers matplotlip<br />
安装clip<br />
cd /home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/CLIP-main<br />
python setup.py build install<br />
安装pyaudio<br />
sudo apt-get install python-dev portaudio19-dev<br />
pip install pyaudio<br />
安装whisper<br />
pip install -U openai-whisper<br />
需要在https://huggingface.co/google-bert/bert-base-uncased/tree/main 中下载pytorch_model.bin，放到send_goals/src/BERT/bert-base-uncased<br />
使用bert.py训练模型，模型命名为train_furniture2.pth<br />
以上两个模型在send_goals/src/nlu.py中调用，可自行修改<br />
启动语义查询<br />
roslaunch send_goals clip_nav.launch<br />
