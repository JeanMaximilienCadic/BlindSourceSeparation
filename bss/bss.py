from bss.gcc_nmf.wav_file import wav_read, wav_write
from bss.gcc_nmf.librosa_stft import stft, istft

import numpy as np
from scipy.signal import argrelmax
from numpy.random import random, seed
import logging
import os

SPEED_OF_SOUND_IN_METRES_PER_SECOND = 340.29

class BSS:
    def __init__(self, method="gcc_nmf", num_sources=2, mic_distance=1.0):
        """

        :param method:
        :param num_sources:
        :param mic_distance:
        """
        # Preprocessing params
        self.method = method
        self.window_size = 1024
        self.fft_size = self.window_size
        self.hop_size = 128
        self.window_function = np.hanning
        self.num_sources=num_sources

        # TDOA params
        self.num_tdoas = 128

        # NMF params
        self.dictionary_size = 128
        self.num_iterations = 100
        self.sparsity_alpha = 0

        # Input params
        self.mic_distance = mic_distance
        self.num_sources = self.num_sources


    def __save_targets__(self, output_dir):
        """

        :param output_dir:
        :return:
        """
        self.num_targets= self.target_signal_estimates.shape[0]
        for target_index in range(self.num_targets):
            source_estimate_filename = "{}/{}_{}.wav".format(output_dir, target_index, name(self.wav_file))
            wav_write(self.target_signal_estimates[target_index], source_estimate_filename, self.sampleRate)

    def __load_mixture_signal__(self, mixture_filename):
        """

        :param mixture_filename:
        :return:
        """
        return wav_read(mixture_filename)

    def __compute_complex_mixture_spectrogram__(self, stereo_samples, fft_size=None):
        """

        :param stereo_samples:
        :param fft_size:
        :return:
        """
        fft_size = self.window_size if fft_size is None else fft_size
        complex_mixture_spectrograms = np.array(
            [stft(np.squeeze(stereo_samples[channel_index]).copy(), self.window_size, self.hop_size, fft_size, np.hanning, center=False)
             for channel_index in np.arange(2)])
        return complex_mixture_spectrograms

    def __get_frequencies_in_hz__(self,  num_frequencies):
        """

        :param num_frequencies:
        :return:
        """
        return np.linspace(0, self.sampleRate / 2, num_frequencies)

    def __get_max_tdoa__(self):
        """
        
        :return: 
        """
        return self.mic_distance / SPEED_OF_SOUND_IN_METRES_PER_SECOND

    def __get_tdoa_as_in_seconds__(self):
        """
        
        :return: 
        """
        maxTDOA = self.__get_max_tdoa__()
        tdoas_in_seconds = np.linspace(-maxTDOA, maxTDOA, self.num_tdoas)
        return tdoas_in_seconds

    def __get_angular_spectrogram__(self, spectral_coherence_v, frequencies_in_hz):
        """

        :param spectral_coherence_v:
        :param frequencies_in_hz:
        :return:
        """
        tdoas_in_seconds = self.__get_tdoa_as_in_seconds__()
        expJOmega = np.exp(np.outer(frequencies_in_hz, -(2j * np.pi) * tdoas_in_seconds))

        FREQ, TIME, TDOA = range(3)
        return np.sum(np.einsum(spectral_coherence_v, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME]).real, axis=1)

    def __estimate_target_tdoa_indexes_from_angular_spectrum__(self, angular_spectrum):
        """

        :param angular_spectrum:
        :return:
        """
        peak_indexes = argrelmax(angular_spectrum)[0]

        logging.info('num_sources provided, taking first %d peaks' % self.num_sources)
        sourcepeak_indexes = peak_indexes[np.argsort(angular_spectrum[peak_indexes])[-self.num_sources:]]

        if len(sourcepeak_indexes) != self.num_sources:
            logging.info(
                'didn''t find enough peaks in ITDFunctions.estimatetarget_tdoa_indexesFromangular_spectrum... aborting')
            os._exit(1)

        # return sources ordered left to right
        sourcepeak_indexes = sorted(sourcepeak_indexes)

        logging.info('Found target TDOAs: %s' % str(sourcepeak_indexes))
        return sourcepeak_indexes

    def __perform_klnmf__(self, v, epsilon=1e-16, seed_value=0):
        """

        :param v:
        :param epsilon:
        :param seed_value:
        :return:
        """
        seed(seed_value)

        w = random((v.shape[0], self.dictionary_size)).astype(np.float32) + epsilon
        H = random((self.dictionary_size, v.shape[1])).astype(np.float32) + epsilon

        for iterationIndex in range(self.num_iterations):
            H *= np.dot(w.T, v / np.dot(w, H)) / (np.sum(w, axis=0)[:, np.newaxis] + self.sparsity_alpha + epsilon)
            w *= np.dot(v / np.dot(w, H), H.T) / np.sum(H, axis=1)

            dictionaryAtomNorms = np.sqrt(np.sum(w ** 2, 0))
            w /= dictionaryAtomNorms
            H *= dictionaryAtomNorms[:, np.newaxis]

        return w, H

    def __get_target_tdoagcnmfs__(self, coherence_v, frequencies_in_hz, target_tdoa_indexes, w, stereo_h):
        """

        :param coherence_v:
        :param frequencies_in_hz:
        :param target_tdoa_indexes:
        :param w:
        :param stereo_h:
        :return:
        """
        self.num_targets= len(target_tdoa_indexes)

        hypothesisTDOAs = self.__get_tdoa_as_in_seconds__()

        num_channels, numAtom, numTime = stereo_h.shape
        normalizedw = w

        expJOmegaTau = np.exp(np.outer(frequencies_in_hz, -(2j * np.pi) * hypothesisTDOAs))

        TIME, FREQ, TDOA, ATOM = range(4)
        target_tdoagcnmfs = np.empty((self.num_targets, numAtom, numTime), np.float32)
        for target_index, targetTDOAIndex in enumerate(target_tdoa_indexes):
            gccChunk = np.einsum(coherence_v, [FREQ, TIME], expJOmegaTau[:, targetTDOAIndex], [FREQ], [FREQ, TIME])
            target_tdoagcnmfs[target_index] = np.einsum(normalizedw, [FREQ, ATOM], gccChunk, [FREQ, TIME],
                                                    [ATOM, TIME]).real

        return target_tdoagcnmfs

    def __get_target_coefficient_masks__(self, target_tdoagcnmfs):
        """

        :param target_tdoagcnmfs:
        :return:
        """
        nanArgMax = np.nanargmax(target_tdoagcnmfs, axis=0)

        target_coefficient_masks = np.zeros_like(target_tdoagcnmfs)
        for target_index in range(self.num_targets):
            target_coefficient_masks[target_index][np.where(nanArgMax == target_index)] = 1
        return target_coefficient_masks

    def __get_target_spectrogram_estimates__(self, target_coefficient_masks, complex_mixture_spectrogram, w, stereo_h):
        """

        :param target_coefficient_masks:
        :param complex_mixture_spectrogram:
        :param w:
        :param stereo_h:
        :return:
        """
        self.num_targets= target_coefficient_masks.shape[0]
        target_spectrogram_estimates = np.zeros((self.num_targets,) + complex_mixture_spectrogram.shape, np.complex64)
        for target_index, targetCoefficientMask in enumerate(target_coefficient_masks):
            for channel_index, coefficients in enumerate(stereo_h):
                target_spectrogram_estimates[target_index, channel_index] = np.dot(w, coefficients * targetCoefficientMask)
        return target_spectrogram_estimates * np.exp(1j * np.angle(complex_mixture_spectrogram))

    def __get_target_signal_estimates__(self, target_spectrogram_estimates):
        """

        :param target_spectrogram_estimates:
        :return:
        """
        num_targets, num_channels, numFreq, numTime = target_spectrogram_estimates.shape
        stftGainFactor = self.hop_size / float(self.window_size) * 2

        target_signal_estimates = []
        for target_index in range(num_targets):
            currentSignalEstimates = []
            for channel_index in range(num_channels):
                currentSignalEstimates.append(
                    istft(target_spectrogram_estimates[target_index, channel_index], self.hop_size, self.window_size, self.window_function))
            target_signal_estimates.append(currentSignalEstimates)
        return np.array(target_signal_estimates) * stftGainFactor

    def separate(self, wav_file, output_dir):
        """

        :param wav_file:
        :param output_dir:
        :return:
        """
        #
        self.wav_file = wav_file
        mixture_filename = self.wav_file

        stereo_samples, self.sampleRate = self.__load_mixture_signal__(mixture_filename)

        complex_mixture_spectrogram = self.__compute_complex_mixture_spectrogram__(stereo_samples)

        num_channels, num_frequencies, numTime = complex_mixture_spectrogram.shape

        frequencies_in_hz = self.__get_frequencies_in_hz__(num_frequencies)

        spectral_coherence_v = complex_mixture_spectrogram[0] * complex_mixture_spectrogram[1].conj() \
                             / abs(complex_mixture_spectrogram[0]) / abs(complex_mixture_spectrogram[1])

        angularSpectrogram = self.__get_angular_spectrogram__(spectral_coherence_v, frequencies_in_hz)

        mean_angular_spectrum = np.mean(angularSpectrogram, axis=-1)

        target_tdoa_indexes = self.__estimate_target_tdoa_indexes_from_angular_spectrum__(mean_angular_spectrum)

        v = np.concatenate(abs(complex_mixture_spectrogram), axis=-1)

        w, H = self.__perform_klnmf__(v)

        num_channels = stereo_samples.shape[0]

        stereo_h = np.array(np.hsplit(H, num_channels))

        target_tdoagcnmfs = self.__get_target_tdoagcnmfs__(spectral_coherence_v,
                                                           frequencies_in_hz,
                                                           target_tdoa_indexes,
                                                           w,
                                                           stereo_h)

        target_coefficient_masks = self.__get_target_coefficient_masks__(target_tdoagcnmfs)

        target_spectrogram_estimates = self.__get_target_spectrogram_estimates__(target_coefficient_masks,
                                                                               complex_mixture_spectrogram,
                                                                               w,
                                                                               stereo_h)

        self.target_signal_estimates = self.__get_target_signal_estimates__(target_spectrogram_estimates)

        self.__save_targets__(output_dir)


if __name__ == "__main__":
    bss = BSS(method="gcc_nmf",
              num_sources=2,
              mic_distance=1.0)
    bss.separate(wav_file="mix.wav",
                 output_dir="./")
