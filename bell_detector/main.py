import logging
import os
from queue import Queue
from threading import Thread
from time import time

from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF

from pushover import init, Client

import sounddevice as sd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class PushWorker(Thread):

    def __init__(self, notif_queue):
        Thread.__init__(self)
        self.notif_queue = notif_queue

        init('<application token here>')
        self.members = [Client("<user token here>")] # Mine

    def run(self):
        while True:
            _ = self.notif_queue.get()
            for client in self.members:
                client.send_message("Deurbel!")


class RecordingWorker(Thread):

    def __init__(self, bell_queue, seconds, fs):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.seconds = seconds
        self.fs = fs

    def run(self):
        while True:
            # Get the work from the bell_queue and expand the tuple
            recording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1)
            sd.wait()
            self.bell_queue.put((self.fs, recording.reshape(recording.shape[0])))


class DetectionWorker(Thread):

    def __init__(self, bell_queue, notif_queue, model_name, model_type):
        Thread.__init__(self)
        self.bell_queue = bell_queue
        self.notif_queue = notif_queue
        self.model_name = model_name
        self.model_type = model_type


    def process_recording(self, signal, fs):
        classifier, mean, std, classes, mid_window, mid_step, short_window, \
            short_step, compute_beat = aT.load_model(self.model_name)

        if fs == 0:
            # audio file IO problem
            return -1, -1, -1
        if signal.shape[0] / float(fs) < mid_window:
            mid_window = signal.shape[0] / float(fs)

        # feature extraction:
        mid_features, s, _ = \
            aF.mid_feature_extraction(signal, fs,
                                    mid_window * fs,
                                    mid_step * fs,
                                    round(fs * short_window),
                                    round(fs * short_step))
        # long term averaging of mid-term statistics
        mid_features = mid_features.mean(axis=1)

        feature_vector = (mid_features - mean) / std    # normalization

        # classification
        class_id, probability = aT.classifier_wrapper(classifier, self.model_type,
                                                      feature_vector)
        return class_id, probability, classes


    def run(self):
        while True:
            fs, recording = self.bell_queue.get()
            start_detection = time()
            try:
                class_out = self.process_recording(recording, fs)[0]
                logging.info(class_out)
                if class_out == 0:
                    logging.info('Sending push notification to queue!')
                    self.notif_queue.put(True)
            except Exception as e:
                print(e)
            finally:
                self.bell_queue.task_done()
            logging.info(f'Elapsed Detection Time: {time() - start_detection}')
            logging.info(f'Queue size: {self.bell_queue.qsize()}')


def main():
    # Create a bell_queue to communicate with the worker threads
    bell_queue = Queue()
    # Create a notif_queue to send push notifications
    notif_queue = Queue()

    # Start processing thread first
    det_worker = DetectionWorker(bell_queue, notif_queue, 'model/svm_bell', 'svm')
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

    # Causes the main thread to wait for the bell_queue to finish processing all the tasks
    bell_queue.join()
    notif_queue.join()

if __name__ == '__main__':
    main()