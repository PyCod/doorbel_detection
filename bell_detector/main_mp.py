import logging
import os
from multiprocessing import Process, Queue
from time import time
from datetime import date

from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF

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
        while True:
            start_recording = time()
            # Get the work from the bell_queue and expand the tuple
            recording = sd.rec(int(self.seconds * self.sr), samplerate=self.sr, channels=1)
            sd.wait()
            self.bell_queue.put((self.sr, recording.reshape(recording.shape[0])))
            logging.info(f'Elapsed Recording Time: {time() - start_recording}')


class DetectionWorker(Process):

    def __init__(self, bell_queue, notif_queue, save_queue,
                 model_name, model_type):
        Process.__init__(self)
        self.bell_queue = bell_queue
        self.notif_queue = notif_queue
        self.save_queue = save_queue
        self.model_name = model_name
        self.model_type = model_type


    def process_recording(self, signal, sr):
        classifier, mean, std, classes, mid_window, mid_step, short_window, \
            short_step, compute_beat = aT.load_model(self.model_name)

        if sr == 0:
            # audio file IO problem
            return -1, -1, -1
        if signal.shape[0] / float(sr) < mid_window:
            mid_window = signal.shape[0] / float(sr)

        # feature extraction:
        mid_features, s, _ = \
            aF.mid_feature_extraction(signal, sr,
                                    mid_window * sr,
                                    mid_step * sr,
                                    round(sr * short_window),
                                    round(sr * short_step))
        # long term averaging of mid-term statistics
        mid_features = mid_features.mean(axis=1)

        feature_vector = (mid_features - mean) / std    # normalization

        # classification
        class_id, probability = aT.classifier_wrapper(classifier, self.model_type,
                                                      feature_vector)
        return class_id, probability, classes


    def run(self):
        while True:
            sr, recording = self.bell_queue.get()
            start_detection = time()
            try:
                class_out = self.process_recording(recording, sr)[0]
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
    det_worker = DetectionWorker(bell_queue, notif_queue, save_queue,
                                 'model/svm_bell', 'svm')
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
