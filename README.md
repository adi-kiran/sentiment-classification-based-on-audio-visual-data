# sentiment-classification-based-on-audio-visual-data
Classification of a person's sentiment based on video (audio+visual) data

The dataset and all the other external linked files can be found at : https://drive.google.com/open?id=1xsXJLdM8sOqMnBW_fiBjdoSoMTSiFfaR

Dataset used is the RAVDESS dataset available at https://zenodo.org/record/1188976
To download complete AV dataset, run the download_dataset.sh file. It consists of videos from 24 actors. For only audio or only video files, visit the link to download.

Files follow the naming convention:
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Multimodal sentiment analysis is a new dimension of the traditional text-based sentiment analysis, which goes beyond the analysis of texts, and includes other modalities such as audio and visual data. It can be bimodal, which includes different combinations of two modalities, or trimodel, which incorporates three modalities. With the extensive amount of social media data available online in different forms such as videos and images, the conventional text-based sentiment analysis has evolved into more complex models of multimodal sentiment analysis, which can be applied in the development of virtual assistants, analysis of YouTube movie reviews, analysis of news videos, and emotion recognition (sometimes known as emotion detection) such as depression monitoring, among others.

Similar to the traditional sentiment analysis, one of the most basic tasks in multimodal sentiment analysis is sentiment classification, which classifies different sentiments into categories such as happy , sad , fearful , surprised , angry neutral etc. Feature engineering, which involves the selection of features that are fed into the deep neural networks, plays a key role in the sentiment classification performance. In multimodal sentiment analysis, a combination of different textual, audio, and visual features are employed.

By combining vocal modulations and facial expressions, it is possible to enrich the feature learning process to better understand affective states of opinion holders. In other words, there could be other behavioral cues in vocal and visual modalities that could be leveraged.
The proposed framework considers both facial landmarks mapping as well as the audio cues that are taken into consideration as features in building the model.
