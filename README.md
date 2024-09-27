# Evasive-HT

How to run the code:

1- pip install -r requirements.txt

2- ipython HW_Trojan_UAP_Generation.py -- --model_name='HTnet' --benchmark='AES-T700' --output_dir='./results/' --sync_epsilon=1.2 --unsync_epsilon=2.4 --resolution=0.1 --sync --unsync --at

For model_name, instead of 'HTnet', you can use 'ResNet-18', 'SVM', 'VGG-11'

Generated patches and models will be stored in the result folder.

