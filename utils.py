import time
import numpy as np


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["embeddings"]


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok

    return d


class Timer:
    def __init__(self):
        self.start_time = {}
        self.running_jobs = []

    def start(self, job):
        if job is None:
            return None

        self.start_time[job] = time.time()
        print("[INFO] {job} started.".format(job=job))
        self.running_jobs.append(job)

    def stop(self, job=None):
        if job is None:
            job = self.running_jobs[-1]

        if job is None or job not in self.start_time:
            return None

        elapsed_time = time.time() - self.start_time[job]
        print("[INFO] {job} finished in {elapsed_time:0.3f}s.".format(job=job, elapsed_time=elapsed_time))
        del self.start_time[job]
        self.running_jobs.remove(job)


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)
