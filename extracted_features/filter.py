import os
import shutil

srcDir = '/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/4-sad/'
targetDir = '/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/only-AV'
for fname in os.listdir(srcDir):
    if not os.path.isdir(os.path.join(srcDir, fname)):
        for prefix in ['01-']:
            if fname.startswith(prefix):
                if not os.path.isdir(os.path.join(targetDir, prefix)):
                    os.mkdir(os.path.join(targetDir, prefix))
                shutil.move(os.path.join(srcDir, fname), targetDir)