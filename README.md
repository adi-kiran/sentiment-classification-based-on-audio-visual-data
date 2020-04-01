# sentiment-classification-based-on-audio-visual-data
Classification of a person's sentiment based on video (audio+visual) data

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

The features (68 facial landmarks and mfcc features) have already been extracted and placed in the extracted_features/csv folder.
