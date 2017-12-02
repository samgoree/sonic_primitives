# sonic_primitives.py
# a recreation of https://github.com/fogleman/primitive/ for sound
# my goal is to read in a .wav file, then use gradient descent to find the sine wave parameters that minimize the error

import wave
import copy
import numpy as np
import sys

TAU = 2 * np.pi
N_ITER = 1000
INITIAL_FREQ_LR = 0.5 # this one is multiplied by the current frequency, since we percieve frequency on a log scale
INITIAL_AMP_LR = 0.2
INITIAL_TIME_LR = 0.5

def choose_max(options, p):
    return options[int(np.argmax(p))]

ITER = 0

def stochastic_in_the_front(options, p):
    if ITER > N_ITER // 2:
        return choose_max(options, p)
    else:
        return np.random.choice(options,p=p)

CHOICE_FCN = stochastic_in_the_front # alternative to np.random.choice


# contains a numpy array for each channel of audio, plus metadata
class Waveform:
    channels = np.array([])
    framerate = 0
    sample_width = 0
    length = 0

    # assumes that data is little-endian, which is usually true of .wav files
    def __init__(self, wave_obj=None):
        if wave_obj is None:
            pass
        else:
            self.sample_width = wave_obj.getsampwidth()
            self.framerate = wave_obj.getframerate()
            self.channels = np.zeros([wave_obj.getnchannels(), wave_obj.getnframes()])
            self.length = len(self.channels[0]) / self.framerate
            for i in range(wave_obj.getnframes()):
                l = list(wave_obj.readframes(1))
                for j,c in enumerate(self.channels):
                    b_array = l[j*self.sample_width:(j+1)*self.sample_width]
                    c[i] = int.from_bytes(b_array, byteorder='little', signed=True)

    def zeros_like(self, wf):
        self.length = wf.length
        self.sample_width = wf.sample_width
        self.framerate = wf.framerate
        self.channels = np.zeros_like(wf.channels)


    # length in seconds, framerate in samples/sec
    def zeros(self, length, sample_width, n_channels, framerate):
        self.sample_width = sample_width
        self.framerate = framerate
        self.legnth = length
        self.channels = np.zeros([n_channels, int(length * framerate)])
    def __add__(self, other):
        assert self.channels[0].shape == other.channels[0].shape
        assert self.framerate == other.framerate
        new_wave = copy.deepcopy(self)
        new_wave.channels += other.channels
        return new_wave

    def __iadd__(self, other):
        assert self.channels[0].shape == other.channels[0].shape
        assert self.framerate == other.framerate
        self.channels += other.channels

    def scalar_difference(self, other):
        assert self.channels[0].shape == other.channels[0].shape
        assert self.framerate == other.framerate 
        return np.sum(np.absolute(np.sum(self.channels, axis=0) - np.sum(other.channels, axis=0)))

    # pretty similar to RMSE
    def scalar_square_difference(self, other):
        assert self.channels[0].shape == other.channels[0].shape
        assert self.framerate == other.framerate 
        return np.sum((np.sum(self.channels, axis=0) - np.sum(other.channels, axis=0))**2)
    # start, end measured in seconds
"""    def subwaveform(self, start,end):
        assert start >= 0
        assert end < self.length
        sub_wave = self()
        sub_wave.zeros(end-start, self.sample_width, self.n_channels, self.framerate)
        sub_wave.channels = self.channels[:,start:end]
        return sub_wave"""



def load_waveform(path):
    w_obj = wave.open(path, 'r')
    return Waveform(w_obj)

def save_waveform(w, path):
    w_obj = wave.open(path, 'w')
    w_obj.setframerate(w.framerate)
    w_obj.setsampwidth(w.sample_width)
    w_obj.setnchannels(len(w.channels))
    for i in range(len(w.channels[0])):
        for j in range(len(w.channels)):
            # we need to keep things within range (clipping, bad)
            w_obj.writeframes(int.to_bytes(
                max(min(
                    int(w.channels[j][i]), 
                    2**(8 * w.sample_width - 1) - 1),
                    -(2**(8 * w.sample_width - 1) - 1)), w.sample_width, byteorder='little', signed=True))

# times in seconds
# amplitudes from -1 to 1
class Sine:
    frequency = 1.0
    amplitude = 0.0
    start_time = 0.0
    end_time = 0.0
    def __init__(self, frequency, amplitude, start_time, end_time):
        assert start_time <= end_time
        self.frequency=frequency
        self.amplitude=amplitude
        self.start_time=start_time
        self.end_time=end_time

    def to_waveform(self, length, sample_width, n_channels, framerate):
        w = Waveform()
        w.zeros(length, sample_width, n_channels, framerate)
        start_frame = int(self.start_time * framerate)
        end_frame = int(self.end_time * framerate)
        amplitude_multiplier = 2**(8 * sample_width -1) -1
        # a sine wave is A * sin(2pi * f * t)
        frames = np.arange(0, end_frame-start_frame)
        w.channels[:, start_frame:end_frame] = self.amplitude * amplitude_multiplier * np.sin(TAU * self.frequency * frames/framerate)
        return w
    def to_waveform_like(self, wf):
        return self.to_waveform(wf.length, wf.sample_width, len(wf.channels), wf.framerate)


def loss_function(current_wf, goal_wf, new_sine_wf):
    new_wf = current_wf + new_sine_wf
    return goal_wf.scalar_square_difference(new_wf)

# this function adjusts a single parameter, either frequency, amplitude or times
# meant for use with python multiprocessing and Pool.map
def adjust_param(param, type):
    pass

# main loop
# create a new amplitude, frequency, start and end time for a sine wave
# measure the difference between the reference waveform and the new one
# use gradient descent for some number of iterations to find the local optimum of that loss function
def main_loop(current_wf, goal_wf):
    
    # create parameters of a new sine wave
    freq = np.random.random() * 10000
    amp = np.random.random() * 2 - 1
    start_time = 0# np.random.random() * current_wf.length
    end_time = current_wf.length # np.random.random() * (current_wf.length - start_time) + start_time
    FREQ_LR = INITIAL_FREQ_LR
    AMP_LR = INITIAL_AMP_LR
    TIME_LR = max(INITIAL_TIME_LR, current_wf.length/10)
    # set the initial loss to be the loss function
    initial_loss = loss_function(current_wf, goal_wf, Sine(freq, 0, start_time, end_time).to_waveform_like(current_wf))
    print("Initial loss: ", initial_loss)
    for i in range(N_ITER):
        global ITER
        ITER = i
        current_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time, end_time).to_waveform_like(current_wf))
        if i % 200 == 0: 
            print("\tIteration", i)
            print("Loss:", current_loss)
            print("Freq:", freq)
            print("Amp:", amp)
            print("Start:", start_time)
            print("End:", end_time)
            FREQ_LR /= 2
            AMP_LR /= 2
            TIME_LR /= 2

        # figure out whether to increment or decrement start time
        delta_time = TIME_LR # np.absolute(np.random.normal(0, TIME_LR))
        # if our start and end time are less than our learning rate apart, don't bother with pos start or neg end loss
        if end_time > start_time + 2 * delta_time:
            pos_start_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time + delta_time, end_time).to_waveform_like(current_wf))
            # also handle times going beyond the start and end of the file
            if start_time - delta_time < 0:
                total = pos_start_loss + current_loss
                start_direction = CHOICE_FCN([1,0], p=[total-pos_start_loss, total - current_loss]/total)
            else: 
                neg_start_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time - delta_time, end_time).to_waveform_like(current_wf))
                total = pos_start_loss + neg_start_loss + current_loss
                start_direction = CHOICE_FCN([-1,0,1], p=[total - neg_start_loss, total - current_loss, total - pos_start_loss]/total/2)
            
            neg_end_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time, end_time - delta_time).to_waveform_like(current_wf))
            if end_time + delta_time >= current_wf.length: 
                total = neg_end_loss + current_loss
                end_direction = CHOICE_FCN([-1,0], p=[total-neg_end_loss, total - current_loss]/total)
            else: 
                pos_end_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time, end_time + delta_time).to_waveform_like(current_wf))
                total = pos_end_loss + neg_end_loss + current_loss
                end_direction = CHOICE_FCN([-1,0,1], p=[total - neg_end_loss, total - current_loss, total - pos_end_loss]/total/2)
        else:
            if start_time - delta_time < 0: start_direction = 0
            else:
                neg_start_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time - delta_time, end_time).to_waveform_like(current_wf))
            
                total = neg_start_loss + current_loss
                start_direction = CHOICE_FCN([-1,0], p=[total-neg_start_loss, total - current_loss]/total)
            if end_time + delta_time >= current_wf.length: end_direction = 0
            else:
                pos_end_loss = loss_function(current_wf, goal_wf, Sine(freq, amp, start_time, end_time + delta_time).to_waveform_like(current_wf))
                total = pos_end_loss + current_loss
                end_direction = CHOICE_FCN([1,0], p=[total-pos_end_loss, total - current_loss]/total)
        
        # figure out whether to increment or decrement frequency
        delta_freq = freq * np.absolute(np.random.normal(0, FREQ_LR))
        pos_freq_loss = loss_function(current_wf, goal_wf, Sine(freq + delta_freq, amp, start_time, end_time).to_waveform_like(current_wf))
        neg_freq_loss = loss_function(current_wf, goal_wf, Sine(freq - delta_freq, amp, start_time, end_time).to_waveform_like(current_wf))
        total = pos_freq_loss + neg_freq_loss + current_loss
        freq_direction = CHOICE_FCN([-1,0,1], p=[total - neg_freq_loss, total - current_loss, total - pos_freq_loss]/total/2)

        # figure out whether to increment or decrement amplitude
        delta_amp = np.absolute(np.random.normal(0, AMP_LR))
        # we don't want amp > 1 or < -1
        if amp + delta_amp > 1:
            neg_amp_loss = loss_function(current_wf, goal_wf, Sine(freq, amp - delta_amp, start_time, end_time).to_waveform_like(current_wf))
            total = neg_amp_loss + current_loss
            amp_direction = CHOICE_FCN([-1,0], p=[total - neg_amp_loss, total - current_loss]/total)
        elif amp - delta_amp < -1:
            pos_amp_loss = loss_function(current_wf, goal_wf, Sine(freq, amp + delta_amp, start_time, end_time).to_waveform_like(current_wf))
            total = pos_amp_loss + current_loss
            amp_direction = CHOICE_FCN([1,0], p=[total - pos_amp_loss, total - current_loss]/total)
        else:
            pos_amp_loss = loss_function(current_wf, goal_wf, Sine(freq, amp + delta_amp, start_time, end_time).to_waveform_like(current_wf))
            neg_amp_loss = loss_function(current_wf, goal_wf, Sine(freq, amp - delta_amp, start_time, end_time).to_waveform_like(current_wf))
            total = pos_amp_loss + neg_amp_loss + current_loss
            amp_direction = CHOICE_FCN([-1,0,1], p=[total - neg_amp_loss, total - current_loss, total - pos_amp_loss]/total/2)

       
        # adjust params
        freq += delta_freq * freq_direction
        amp += delta_amp * amp_direction
        start_time += delta_time * start_direction
        end_time += delta_time * end_direction
    # update current_wf
    if current_loss < initial_loss:
        current_wf += Sine(freq, amp, start_time, end_time).to_waveform_like(current_wf)
    else:
        print("Loss increased significantly, not using that wave.")
            


if __name__=='__main__':
    w = load_waveform(sys.argv[1])
    new_w = Waveform()
    new_w.zeros_like(w)
    for i in range(1, 1001):
        print("Wave", i)
        main_loop(new_w, w)
        if i%20 == 0:
            save_waveform(new_w, sys.argv[2] + str(i) + '.wav')