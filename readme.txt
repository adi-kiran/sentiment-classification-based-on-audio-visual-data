Project Title : Multimodel Sentiment Analysis based on Audio and Visual feed
Team : Adithya Kiran (PES1201700231), Chirag P Tubakad (PES1201700896)

<---------- Dependent files ---------->
All the necesary external linked files can be found at :  https://drive.google.com/open?id=1xsXJLdM8sOqMnBW_fiBjdoSoMTSiFfaR

<---------- General Instructions ----------> 
> All dependencies are mentioned in dependencies.txt
> All of the libraries are needed to run main.py, which trains our model, displays test accuracy and other metrics.
> The files in saved_models, is a model which we have already trained.
> The 8-emotions-tf-predict.py.py, is a file that can be run to test our saved_model on a video.
> To run 8-emotions-tf-predict.py, we need imutils, librosa, dlib and cv2 libraries installed, for feature extraction from the video.
> Unzip the shape_predictor_68_face_landmarks.zip file before running predict.py
> Find the training dataset at : 
> To download a sample video to try out 8-emotions-tf-predict.py, download test.mp4 from the same drive link.

<---------- Specific Instructions ---------->
2-emotion-keras.py : Contains the keras model used to build the binary classification of emotions.
2-emotion-tf.py : Contains the tensorflow version of the model for binary classification of emotions.
8-emotions-keras.py : Contains the keras code for the 8 emotions classification model.
8-emotions-tf-main.py : Contains the tensorflow version of the model for classification of 8 emotions.