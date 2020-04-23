import subprocess as sbp
import os

path= "/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/3-happy"
fol = os.listdir(path)
p2 = "/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/4-sad"

for i in fol:
    p1 = os.path.join(path,i)
    p3 = 'cp -r ' + p1 +' ' + p2+'/.'
    sbp.Popen(p3,shell=True)