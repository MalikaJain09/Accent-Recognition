import pandas as pd

from tqdm import tqdm    
import soundfile as sf
import matplotlib.pyplot as plt
import math

from tensorflow.keras.utils import plot_model
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
window_size = 64
window_hop = 30
EPOCHS = 250 #35#250

def resampled_wav(df, wav_audio,resampling_data ):  
    ind = df.index[df['language_num'] == wav_audio].tolist()
    if len(ind)>0:
        sf.write('Resampled Audio/{}.wav'.format(wa_audio), resampling_data['Resampling_Data'][ind[0]], samplerate = RATE)
        print('{}.wav saved'.format(wav_audio))
    else:
        print('Audio not found')


def get_wav(wav_df):
    '''
    Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate
    y: audio_timeseries: array
    sr = sampling rate: 44100
    '''
    data = pd.DataFrame([])
    for i in tqdm(range(wav_df.shape[0])):
        try:
            y, sr = librosa.load('../audio/{}.wav'.format(wav_df['language_num'][i]))
            resample= librosa.core.resample(y=y,orig_sr=sr , target_sr=RATE, scale=True)
            data = data.append({'Resampling_Data':resample.tolist()},ignore_index=True)      
        except Exception as e:
            print('ERROR',e)
        
    return data
   
 
def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    data = pd.DataFrame([])
    for i in tqdm(range(wav.shape[0])):
        try:    
            mfcc= librosa.feature.mfcc(y=np.array(wav['Resampling_Data'][i]), sr=RATE, n_mfcc=N_MFCC)
            data = data.append({'MFCC_Data':mfcc},ignore_index=True) 
        
        except Exception as e:
            print('ERROR',e)
        
    return data

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    
    for mfcc, label in zip(X_mfcc_train['MFCC_Data'], labels):
        start_frame = window_size 
        end_frame = window_hop + math.floor(float(mfcc.shape[1]) / window_hop)
        
        for frame_idx in range(start_frame, end_frame, window_hop):
            window = mfcc[:, frame_idx-window_size:frame_idx]
            segments.append(window)
            seg_labels.append(label)
            
    return(segments, seg_labels)
    
def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    start_frame = window_size 
    end_frame = window_hop * math.floor(float(mfcc.shape[1]) / window_hop)
        
    segments = []
    for frame_idx in range(start_frame, end_frame, window_hop):
            window = mfcc[:, frame_idx-window_size:frame_idx]
            segments.append(window)

    return(np.array(segments))


def create_segmented_mfccs(X):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X['MFCC_Data']:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def train_model(X_train,y_train,X_validation,y_validation, image_name, batch_size=128): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''
    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)
    
    model = Sequential()
    #(17786, 13, 64, 1)  Grey scale
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
    
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    # es = EarlyStopping(monitor='accuracy', min_delta=.005, patience=10, verbose=1, mode='auto')
    # lr = ReduceLROnPlateau(monitor='val_accuracy',patience=3, verbose=1, factor=0.5, 
                                            # min_lr=0.00001)
    batch_size=128
    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='logs', histogram_freq=1)

    plot_model(model, to_file='../models/{}.png'.format(image_name), show_shapes=True, show_layer_names=True)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        steps_per_epoch=len(X_train) / batch_size
                        , epochs=EPOCHS,
                        callbacks=[es,tb,lr], validation_data=(X_validation,y_validation))

    return (history, model)


def save_model(model, model_filename):
    model.save('../models/CNNLayer_model.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'
    
    
def most_frequent(List):
    return max(set(List), key = List.count)


def predict_class(audio_data,model,labels):
    audio_data= create_segmented_mfccs(audio_data)
    prediction_list=[]
    for i in audio_data:
        shape= i.shape
        test_audio = i.reshape(shape[0], shape[1], shape[2], 1 )        
        y_predicted = model.predict_classes(test_audio)
        prediction_list.append(label[most_frequent(y_predicted.tolist())])
    return prediction_list


def evaluation_Matrices(y_actual, y_predict):
    prediction_label=pd.DataFrame(data=np.array(y_predict), columns= ['native_language'])
    prediction_label = pd.get_dummies(prediction_label, columns = ['native_language'])
    
    y_actual = y_actual.values.argmax(axis=1)
    prediction_label= prediction_label.values.argmax(axis=1)
    
    cm_test= confusion_matrix(y_actual, prediction_label)
    print(cm_test)
    print(precision_score(y_actual, prediction_label, average = 'macro'))
    print(recall_score(y_actual, prediction_label, average = 'macro'))
    print(f1_score(y_actual, prediction_label, average = 'macro'))


def loss_accuracy(x, y,model):
    rows = x[0].shape[0]
    cols = x[0].shape[1]
    x = x.reshape(x.shape[0], rows, cols, 1 )
    loss, accuracy = model.evaluate(x, y, verbose=1)


def plot_model_history(model_history, image_name):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['validation_accuracy'])+1),model_history.history['validation_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['validation_loss'])+1),model_history.history['validation_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('../models/CNNLayer_model_history.png'.format(image_name))

    
if __name__ == '__main__':
    
    #step1: Load Data
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train= pd.read_csv('y_train.csv')
    y_test= pd.read_csv('y_test.csv')
    
    y_train_class=[]
    label ={0:'Arabic',1:'English',2:'Mandarin'}
    for i in range(len(y_train)) :
        y_train_class.append(label[np.argmax(y_train.loc[i])])
        
    y_test_class=[]
    label ={0:'Arabic',1:'English',2:'Mandarin'}
    for i in range(len(y_test)) :
        y_test_class.append(label[np.argmax(y_test.loc[i])])
    
    
    #Step2: Get resampled wav files
    X_resampling_train = get_wav(pd.DataFrame(X_train['language_num']))
    X_resampling_test = get_wav(pd.DataFrame(X_test['language_num']))
    
    resampled_wav(X_train , 'arabic180', X_resampling_train )
    resampled_wav(X_train , 'english595', X_resampling_train )
    resampled_wav(X_train , 'mandarin127', X_resampling_train )
    
    
    #Step3: Convert to MFCC
    X_mfcc_train = to_mfcc(X_resampling_train)
    X_mfcc_test = to_mfcc(X_resampling_test)
    
      # Step4:Create segments from MFCCs
    X_segment_train, y_segment_train = make_segments(X_mfcc_train, y_train_class)
    X_validation, y_validation = make_segments(X_mfcc_test, y_test_class)
    
    
    # Step5:Onehot encodding of segment labels
    y_segment_train=pd.DataFrame(data=np.array(y_segment_train), columns= ['native_language'])
    y_segment_train = pd.get_dummies(y_segment_train, columns = ['native_language'])
    
    y_validation=pd.DataFrame(data=np.array(y_validation), columns= ['native_language'])
    y_validation = pd.get_dummies(y_validation, columns = ['native_language'])
   
    #-------------------------------------------------------------------------------  

    # Step6:Train model
    history, model = train_model( np.array(X_segment_train), np.array(y_segment_train), 
                                 np.array(X_validation),np.array(y_validation), image_name)

    # Step7: Save model
    save_model(model, 'CNNLayer_model6'.format(image_name))
    
    # Step8:Prediction
    prediction = predict_class(X_mfcc_test,model,label)
    
    train_predict = predict_class(X_mfcc_train,model,label)
    
    # Step9:Evaluation Matrices
    evaluation_Matrices(y_train, train_predict)
    evaluation_Matrices(y_test, prediction)
    
    loss_accuracy(np.array(X_validation), np.array(y_validation), model)
    loss_accuracy(np.array(X_segment_train), np.array(y_segment_train), model)
    
    plot_model_history(history, image_name)



