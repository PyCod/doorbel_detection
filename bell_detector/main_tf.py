import logging
import os
from queue import Queue
from threading import Thread
from time import time, sleep
from datetime import datetime

import librosa
import numpy as np
import requests
from tflite_runtime.interpreter import Interpreter

from pushover import init, Client

import sounddevice as sd
import soundfile as sf

from creds import app_token, client_token

sd.default.device = 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class PushWorker(Thread):

    def __init__(self, notif_queue):
        Thread.__init__(self)
        self.notif_queue = notif_queue

        init(app_token)
        self.members = [Client(client_token)] # Mine

        self.last_pinged = time()

    def run(self):
        while True:
            datetimestr = self.notif_queue.get()
            if time() - self.last_pinged > 5:
                for client in self.members:
                    client.send_message(f'http://192.168.0.128:8080/{datetimestr}.wav')
                requests.get('http://192.168.0.206/bell')
                self.last_pinged = time()
            self.notif_queue.task_done()


class SaveWorker(Thread):

    def __init__(self, save_queue, location='triggers', sr=22050):
        Thread.__init__(self)
        self.save_queue = save_queue
        self.location = location
        self.sr = sr

    def run(self):
        while True:
            recording, datetimestr = self.save_queue.get()
            sf.write(os.path.join(self.location, f'{datetimestr}.wav'),
                     recording, self.sr, subtype='PCM_24')
            self.save_queue.task_done()


class RecordingWorker(Thread):

    def __init__(self, bell_queue, seconds, sr):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.seconds = seconds
        self.sr = sr

    def run(self):
        prev_time = time()
        while True:
            # logging.info(f'Elapsed Time Between recordings: {time() - prev_time}')
            # Get the work from the bell_queue and expand the tuple
            recording = sd.rec(int(self.seconds * self.sr), samplerate=self.sr, channels=1)
            sd.wait()
            self.bell_queue.put((self.sr, recording.reshape(recording.shape[0])))
            prev_time = time()


class DetectionWorker(Thread):

    def __init__(self, bell_queue, notif_queue, save_queue):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.notif_queue = notif_queue
        self.save_queue = save_queue
        # Load the TFLite model and allocate tensors.
        self.interpreter = Interpreter(model_path="model/bell_model_187.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def process_recording(self, signal, sr):

        if sr == 0:
            # audio file IO problem
            return -1, -1, -1
        X = signal.T

        # logging.info(f'{X.shape}, {np.mean(X)}, {np.std(X)}')

        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
        # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T,axis=0)

        ext_features = np.hstack([mfccs, chroma, mel, contrast])

        ext_features = np.expand_dims(ext_features, axis=0)
        ext_features = np.expand_dims(ext_features, axis=2)
        ext_features = ext_features.astype(np.float32)

        # logging.info(f'{ext_features.shape}, {np.mean(ext_features)}, {np.std(ext_features)}')

        # classification
        self.interpreter.set_tensor(self.input_details[0]['index'], ext_features)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])

        # logging.info(f'Pred: {pred}')

        return round(pred[0][0]), pred[0][0]


    def run(self):
        while True:
            sr, recording = self.bell_queue.get()
            start_detection = time()
            try:
                class_out, prob = self.process_recording(recording, sr)
                logging.info(f'{class_out} - {prob}')
                datetimestr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                if class_out == 0:
                    logging.info('Sending push notification to queue!')
                    self.notif_queue.put((datetimestr))
                    self.save_queue.put((recording, datetimestr))
            except Exception as e:
                print(e)
            finally:
                self.bell_queue.task_done()
            # logging.info(f'Elapsed Detection Time: {time() - start_detection}')
            # logging.info(f'Queue size: {self.bell_queue.qsize()}')


def main():
    # Create a bell_queue to communicate with the worker threads
    bell_queue = Queue()
    # Create a notif_queue to send push notifications
    notif_queue = Queue()
    # Create a save_queue to save sounds for later reuse in training
    save_queue = Queue()

    # Start processing thread first
    det_worker = DetectionWorker(bell_queue, notif_queue, save_queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    det_worker.daemon = False
    det_worker.start()

    # Start recording
    rec_worker = RecordingWorker(bell_queue, 0.5, 22050)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    rec_worker.daemon = False
    rec_worker.start()

    # Start the push notification listener
    notif_worker = PushWorker(notif_queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    notif_worker.daemon = False
    notif_worker.start()

    # Start the saving worker which will save all bell instances
    save_worker = SaveWorker(save_queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    save_worker.daemon = False
    save_worker.start()

    # Causes the main thread to wait for the bell_queue to finish processing all the tasks
    bell_queue.join()
    notif_queue.join()

if __name__ == '__main__':
    main()
