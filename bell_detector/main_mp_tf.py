import logging
import os
from multiprocessing import Process, Queue
from time import time
from datetime import date

import librosa
import numpy as np
import tflite_runtime.interpreter as tflite

from pushover import init, Client

import soundfile as sf

from creds import app_token, client_token


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class PushWorker(Process):

    def __init__(self, notif_queue):
        Process.__init__(self)
        self.notif_queue = notif_queue

        init(app_token)
        self.members = [Client(client_token)] # Mine

        self.last_pinged = time()

    def run(self):
        while True:
            _ = self.notif_queue.get()
            if time() - self.last_pinged > 5:
                for client in self.members:
                    client.send_message("Deurbel!")
                self.last_pinged = time()
            # self.notif_queue.task_done()


class SaveWorker(Process):

    def __init__(self, save_queue, location='triggers', sr=22050):
        Process.__init__(self)
        self.save_queue = save_queue
        self.location = location
        self.sr = sr

    def run(self):
        while True:
            recording = self.save_queue.get()
            sf.write(os.path.join(self.location, f'{date.today()}.wav'),
                     recording, self.sr, subtype='PCM_24')
            # self.save_queue.task_done()


class RecordingWorker(Process):

    def __init__(self, bell_queue, seconds, sr):
        Process.__init__(self)
        self.bell_queue = bell_queue
        self.seconds = seconds
        self.sr = sr

    def run(self):
        import sounddevice as sd
        prev_time = time()
        while True:
            logging.info(f'Elapsed Time Between recordings: {time() - prev_time}')
            # Get the work from the bell_queue and expand the tuple
            recording = sd.rec(int(self.seconds * self.sr), samplerate=self.sr, channels=1)
            sd.wait()
            self.bell_queue.put((self.sr, recording.reshape(recording.shape[0])))
            prev_time = time()


class DetectionWorker(Process):

    def __init__(self, bell_queue, notif_queue, save_queue):
        Process.__init__(self)
        self.bell_queue = bell_queue
        self.notif_queue = notif_queue
        self.save_queue = save_queue
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path="model/bell_model_fp16.tflite", num_threads=4)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def process_recording(self, signal, sr):

        if sr == 0:
            # audio file IO problem
            return -1, -1, -1
        X = signal.T

        start = time()

        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T,axis=0)

        print(f'Features: {time() - start}')

        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        ext_features = np.expand_dims(ext_features, axis=0)
        ext_features = np.expand_dims(ext_features, axis=2)
        ext_features = ext_features.astype(np.float32)

        # classification
        self.interpreter.set_tensor(self.input_details[0]['index'], ext_features)
        start = time()
        self.interpreter.invoke()
        print(f'Interpreter: {time() - start}')
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(pred)

        return round(pred[0][0])


    def run(self):
        while True:
            sr, recording = self.bell_queue.get()
            start_detection = time()
            try:
                class_out = self.process_recording(recording, sr)
                logging.info(class_out)
                if class_out == 0:
                    logging.info('Sending push notification to queue!')
                    self.notif_queue.put(True)
                    self.save_queue.put(recording)
            except Exception as e:
                print(e)
            finally:
                # self.bell_queue.task_done()
                pass
            logging.info(f'Elapsed Detection Time: {time() - start_detection}')
            logging.info(f'Queue size: {self.bell_queue.qsize()}')


def main():
    # Create a bell_queue to communicate with the worker Processs
    bell_queue = Queue()
    # Create a notif_queue to send push notifications
    notif_queue = Queue()
    # Create a save_queue to save sounds for later reuse in training
    save_queue = Queue()

    # Start processing Process first
    det_worker = DetectionWorker(bell_queue, notif_queue, save_queue)
    # Setting daemon to True will let the main Process exit even though the workers are blocking
    det_worker.daemon = True
    det_worker.start()

    # Start recording
    rec_worker = RecordingWorker(bell_queue, 0.5, 22050)
    # Setting daemon to True will let the main Process exit even though the workers are blocking
    rec_worker.daemon = True
    rec_worker.start()

    # Start the push notification listener
    notif_worker = PushWorker(notif_queue)
    # Setting daemon to True will let the main Process exit even though the workers are blocking
    notif_worker.daemon = True
    notif_worker.start()

    # Start the saving worker which will save all bell instances
    save_worker = SaveWorker(save_queue)
    # Setting daemon to True will let the main Process exit even though the workers are blocking
    save_worker.daemon = True
    save_worker.start()

    # Causes the main Process to wait for the bell_queue to finish processing all the tasks
    det_worker.join()
    rec_worker.join()
    notif_worker.join()
    save_worker.join()

if __name__ == '__main__':
    main()
