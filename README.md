# dogGAN

輸入動物影片，然後輸出卡通化的動物影片。
使用物件辨識、影像切割以及GAN的轉換風格，訓練出理想中的風格轉化，之後藉由影像拼接來達成目的。
朱庭君 00857007 
林雅芸 00857011
姜吳亭 00857025
劉芷辰 00857101

用法: 
先安裝conda, python, git

git clone https://github.com/liuliu0099/dogGAN.git
cd ./dogGAN
conda env create --name dogGAN --file ./environments/conda_env_GPU.yaml
conda activate dogGAN
python test.py --input_video ./sample/input.mp4 --output_video ./sample/output.mp4
