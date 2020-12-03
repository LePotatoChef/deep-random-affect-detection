# Affect Detection Using (Random) Deep Networks
### CS/RBE 539 Team 7 
### Sami Baral, Zekun Dai, Jessica Herman, Hanwen Hong, Ethan Prihar

## Background and Motivation

Affect detection is one of the hot topics in the learning science and technology field [2]. The task of affect detection is to predict the emotional state of a student as they complete an assignment(student affect).  Within the context of intelligent tutoring systems, which are designed to be deployed at scale, the use of physical and physiological sensors to measure affect is not practical.  
Past studies have shown that affect correlates with student performance and as a result, there have been previous attempts to predict student affect from their actions within online tutoring platforms. Predicting student affect can be used to alert teachers to frustrated or bored students, allowing them to intervene and get the student back on task.  ASSISTments is an online learning platform developed here at WPI which three of our five team members are involved with.  The ASSISTments team is developing ways to use affect models to report student’s emotional states live to teachers [1]. This could potentially help teachers who are teaching their students remotely due to COVID19 situation understand the emotional states of their students even though they aren’t in the same room as them.
Affect detectors exist, but they require the extraction of features from the action level data.  This feature extraction process is costly and time consuming, which impedes its ability to be used in real time applications like live classroom interventions. Additionally, most sensor-free affect detection models are trained using simple regression models, decision trees, or from shallow neural networks. These simpler models do not predict affect with sufficient accuracy.  Deep learning techniques are just starting to be investigated.  It is hoped that the success of these complex models in other areas can be realized in this context, but training time and and data requirements present challenges to their use.  

The purpose of our project is to design a model to predict student affect more robustly and efficiently based solely on interactions with a tutoring system. 

We explored a claim in a recent paper[3], motivated by Cover's Theorem[4] that any sufficiently large LSTM doesn’t require training except in the output layer. This could allow us to reap the benefits of using a very large LSTM network while essentially only needing to train a logistic regression in the last layer. If successful, such a deep affect detector in areas that haven’t had enough data to train deep learning models through traditional techniques. 

## Methodology

We examined existing affect detectors, and identified the data that the detectors were trained with.  The data used to train previously published affect detection models were collected from ASSISTments which is available to us. We did the necessary data cleaning and get it ready for the model training and evaluation.
We designed and implemented XXX models using available tools from {keras, ...?}
We trained each model with the data we collected from ASSISTments in the first step. For the LSTM, we both fully trained the network for use as a control, and only trained the last layer for our model under test.  **Need more here on how we did the training.

We demonstrate the utility of our model by comparing the roc-auc, Kappa, and training time of our method to the previously published methods. If we can achieve higher roc-auc and Kappa scores, or if we can achieve faster training time, we will consider this method a success. 

## Results

### Plots
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/bilstm_expert_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/bilstm_expert_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/bilstm_raw_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/bilstm_raw_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/borep_expert_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/borep_expert_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/borep_raw_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/borep_raw_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/esn_expert_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/esn_expert_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/esn_raw_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/esn_raw_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/lstm_expert_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/lstm_expert_kappa.png" width="400" height="300">
</p>
<p float="left">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/lstm_raw_auc.png" width="400" height="300">
<img src="https://github.com/Ethanprihar/deep-random-affect-detection/blob/main/ethan/clean/plots/random/lstm_raw_kappa.png" width="400" height="300">
</p>

## References

[1]Botelho, Anthony & Baker, Ryan & Heffernan, Neil. (2018). Improving Sensor-Free Affect Detection Using Deep Learning. 10.1007/978-3-319-61425-0_4.

[2]Pardos, Z.A., Baker, R.S.J.d., San Pedro, M.O.C.Z., Gowda, S.M., Gowda, S.M. (2013) Affective states and state tests: Investigating how affect throughout the school year predicts end of year learning outcomes. Proceedings of the 3rd International Conference on Learning Analytics and Knowledge, 117-124

[3]Wieting, J., & Kiela, D. (2019). No Training Required: Exploring Random Encoders for Sentence Classification. ArXiv, abs/1901.10444.

[4]T. M. Cover, "Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition," in IEEE Transactions on Electronic Computers, vol. EC-14, no. 3, pp. 326-334, June 1965, doi: 10.1109/PGEC.1965.264137.
