import collections
import contextlib
import wave
import webrtcvad
import numpy as np
from sklearn import svm
from python_speech_features import mfcc
from sklearn.decomposition import PCA

frame_duration_ms = 10
padding_duration_ms = 50

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = wf.getparams()
        assert nchannels == 1
        assert sampwidth == 2
        assert framerate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(nframes)
        types = {
            1: np.int8,
            2: np.int16,
            4: np.int32
        }
        samples = np.frombuffer(pcm_data, dtype = types[sampwidth])
        return samples, framerate

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    list_out = []
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen = num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                list_out += (f.bytes for f in voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        list_out += (f.bytes for f in voiced_frames)
    return list_out

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def VAD(audio, sample_rate):
    vad = webrtcvad.Vad(3)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    audio_after_vad = vad_collector(sample_rate, frame_duration_ms,
                                    padding_duration_ms, vad, frames)
    return np.array(audio_after_vad).flatten()

def PCA_(audio_after_mfcc):
    pca = PCA(n_components=13)
    pca.fit(audio_after_mfcc.T)
    '''print(pca.explained_variance_ratio_)

    print("pca.shape: ", pca.components_.shape)'''

    # Количество компонент и доля их участия в объяснении зависимостей
    sum = 0
    for idx, explained_variance in enumerate(pca.explained_variance_ratio_):
        sum += explained_variance
        # print(idx + 1, sum)

    pca = PCA(n_components=8)  # 8 компонент содержат в себе 96% информации (дисперсии)
    pca.fit(audio_after_mfcc.T)
    return pca.components_

audio, sample_rate = read_wave("Voices/Anna/Anna (1).wav")
audio_after_vad = VAD(audio, sample_rate)
audio_after_mfcc = mfcc(audio_after_vad, sample_rate)
audio_after_PCA = PCA_(audio_after_mfcc)

audio2, sample_rate2 = read_wave("Voices/Dima/Dima (1).wav")
audio_after_vad2 = VAD(audio2, sample_rate2)
audio_after_mfcc2 = mfcc(audio_after_vad2, sample_rate2)
audio_after_PCA2 = PCA_(audio_after_mfcc2)

audio3, sample_rate3 = read_wave("Voices/Anna/Anna (2).wav")
audio_after_vad3 = VAD(audio3, sample_rate3)
audio_after_mfcc3 = mfcc(audio_after_vad3, sample_rate3)
audio_after_PCA3 = PCA_(audio_after_mfcc3)

audio4, sample_rate4 = read_wave("Voices/Dima/Dima (2).wav")
audio_after_vad4 = VAD(audio4, sample_rate4)
audio_after_mfcc4 = mfcc(audio_after_vad4, sample_rate4)
audio_after_PCA4 = PCA_(audio_after_mfcc4)

# Данные для обучения
X_train = np.concatenate((audio_after_PCA.T, audio_after_PCA2.T))
y_train = np.concatenate((np.ones(audio_after_PCA.T.shape[0]),
                          np.zeros(audio_after_PCA2.T.shape[0])))
clf = svm.SVC()
clf.fit(X_train, y_train)

X_test = audio_after_PCA4.T
y_test = np.ones(audio_after_PCA4.T.shape[0])

print(clf.score(X_test, y_test))