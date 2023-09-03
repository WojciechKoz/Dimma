# Dimma: Semi-supervised Low Light Image Enhancement with Adaptive Dimming

To run this code install requirements 
```bash
pip install -r requirements.txt
```
and run one of the following commands:

```bash
python train_supervised.py
python finetune.py
```
For different config file use --config flag. There are many configs in config folder.

Please, bear in mind that you need to first train unsupervised model before running finetune.py. Data and models are not included in this repository. You can get them from the following link: [drive](https://drive.google.com/drive/folders/1mobXx1HI8BS-C8_-U-EHNvbMEPCoGIUK?usp=sharing).
