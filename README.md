# Time Series Prediction with Time Delay Neural Network

This is a PyTorch implementation of the Time Delay Neural Network architecture from [[1]](#1) applied to predicting Mackey-Glass time series data. TDNN architectures were a precursor to Recurrent Neural Networks and are still actively researched for speech recognition and time series prediction. The input to the network is ten previous samples and the network predicts the next sample. 

![](https://github.com/btilmon/TDNN/blob/master/figs/Figure_1.png)



#### References
<a id="1">[1]</a> 
A. Waibel, T. Hanazawa, G. Hinton, K. Shikano, K. Lang. 
"Phoneme Recognition using Time-Delay Neural Networks".
IEEE Transactions on Acoustics, Speech, and Signal Processing. 1989.

