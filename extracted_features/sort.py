import os, shutil

os.chdir("/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/")

for f in os.listdir("csv"):
    folderName = f[7:9]

    if not os.path.exists(folderName):
        os.mkdir(folderName)
        shutil.copy(os.path.join('csv', f), folderName)
    else:
        shutil.copy(os.path.join('csv', f), folderName)