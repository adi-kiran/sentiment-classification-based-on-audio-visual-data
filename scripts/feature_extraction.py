import dlib, imutils, subprocess, cv2, time, librosa
from imutils.video import FileVideoStream
from imutils import face_utils
from os import walk
import pandas as pd
import numpy as np

input_folder = '../../dataset/'
output_folder = '../extracted_features/'
count = 0
files_list=[]
mypath = "/home/adithya/projects/deep_learning/dataset"
for (dirpath, dirnames, filenames) in walk(mypath):
    files_list.extend(filenames)
files_list.sort()
for file in files_list:
        print(file)
        input_file = input_folder+file
        output_file_wav = output_folder+"wav/"+file.split(".")[0]
        output_file_csv = output_folder+"csv/"+file.split(".")[0]
        mode,channel,emotion,intensity,statement,repetition,actor = file.split("-")
        actor = actor.split(".")[0]
        # List containing numpy arrays for each frame for the current dataset
        features_list = []
        # initialize dlib's HOG based face detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # video stream thread
        vs = FileVideoStream(input_file).start()
        fileStream = True
        # extracting audio from video
        command = 'ffmpeg -i ' + input_file + ' -ab 160k -ac 2 -ar 44100 -vn ' + output_file_wav + '.wav 2>abc.txt'
        subprocess.call(command, shell = True)
        # initialising fps, frame count, duration and timeout
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)      
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count // fps
        timeout = time.time() + duration
        # capturing frame data (68 (x,y) facial data points)
        try:
                while time.time() <= timeout:
                        # reading in one frame at a time
                        frame = vs.read()
                        frame = imutils.resize(frame, width = 400)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        rects = detector(gray, 0)
                        for rect in rects:
                                # determine the facial landmarks for the face region, then convert the facial landmark (x, y) coordinates to a numpy array
                                shape = predictor(gray, rect)
                                shape = face_utils.shape_to_np(shape)
                        	# loop over the (x, y) coordinates for the facial landmarks
                                # list_temp contains [1_x, 1_y, .... , 68_x, 68_y]
                                list_temp = []
                                for (x, y) in shape:
                                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                                        list_temp.extend([x,y])
                                # For each frame, append to a list so we have a list of lists where each inner list contains 136 values i.e. [1_x, 1_y, .... , 68_x, 68_y]
                                features_list.append(list_temp)
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
                cv2.destroyAllWindows()
                vs.stop()
        except:
                cv2.destroyAllWindows()
                vs.stop()
                print("error")
        # creating a pandas dataframe with 137 columns where first 136 columns will have [1_x, 1_y, .... , 68_x, 68_y] and the 137th column will have the truth_value
        df = pd.DataFrame(features_list)
        # creating column names for the dataframe
        columns = []
        for i in range(1,69):
                for j in ['X', 'Y']:
                        columns.append(str(i)+j)
        df.columns = columns
        # adding emotion, intensity, statement and actor to the dataset 
        df['emotion'] = [int(emotion) for i in features_list]
        df['intensity'] = [int(intensity) for i in features_list]
        df['statement'] = [int(statement) for i in features_list]
        df['actor'] = [int(actor) for i in features_list]       
        # extracting audio features
        features = pd.DataFrame(columns=['feature'])
        data, sampling_rate = librosa.load(output_file_wav + '.wav')
        X, sample_rate = librosa.load(output_file_wav + '.wav', res_type = 'kaiser_fast', sr = 22050*2, offset = 0)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 20).T, axis=0)
        feature = mfccs
        features.loc[0] = [feature]
        features = pd.DataFrame(features.feature.tolist(), columns = [i for i in range(20)])
        df.append([features], ignore_index = True)
        avg = features.mean(axis = 1)
        for i in range(20):
            df[str(i+1)+'A'] = feature[i]
        # adding mean_MFCC to the facial landmarks dataset
        df['mean_MFCC'] = list(avg)[0]
        # saving training dataset
        df.to_csv(output_file_csv+'.csv', header = True, index = False)
        count+=1
        print("DONE",count)