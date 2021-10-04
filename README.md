# Violence-Detection-CNN-LSTM inference
Violence Detection using pre-trained CNN and LSTM


## Prerequisites
```bash
pip install -r requirements.txt
```

## How to use
You can download the pre-trained network from [here](https://drive.google.com/drive/folders/1nRcNmFYkrkG4d5UYQg75j6ZDzi5Gw4oN?usp=sharing).
The files should be placed in project directory.
The input video should be located in /video folder with it's filename of 'vi[number].mp4' or 'no[number].mp4'. You can download sample videos in [here](https://drive.google.com/drive/folders/1PcerEu2eLJigjeBUmm5zHHTa7fS3_1W7?usp=sharing).
Then, run
```bash
python violence_inference.py
```

Possible outputs: 

```bash
Probability of Violence of vi7.mp4 is 95.665598 %
Probability of Violence of no3.mp4 is 7.113001 %
Probability of Violence of vi1.mp4 is 29.015443 %
Probability of Violence of vi6.mp4 is 97.666192 %
Probability of Violence of no4.mp4 is 96.412838 %
Probability of Violence of no2.mp4 is 6.056682 %
Probability of Violence of vi4.mp4 is 97.660011 %
Probability of Violence of no1.mp4 is 6.312908 %
Probability of Violence of vi9.mp4 is 96.783108 %
Probability of Violence of vi8.mp4 is 97.333872 %
Probability of Violence of vi3.mp4 is 97.467607 %
Probability of Violence of no5.mp4 is 97.627854 %
Probability of Violence of vi2.mp4 is 96.353370 %
Probability of Violence of no7.mp4 is 86.571866 %
Probability of Violence of no6.mp4 is 5.311197 %
Probability of Violence of vi5.mp4 is 96.849132 %

```

