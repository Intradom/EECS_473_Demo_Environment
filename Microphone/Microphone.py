import pyaudio
import numpy as np

# Mic settings
PYAUDIO_FORMAT = pyaudio.paInt32
NUMPY_FORMAT = np.int32
CHANNELS = 1
RATE = 44100

# Paramters
COLLECT_FREQUENCY = 2.5 # in Hertz
VALUES_TO_AVERAGE = 1
MIN_FREQ = 80 # ignore freq values below this
MAX_FREQ = 260 # ignore freq values above this
CAL_TIME_EACH_COMMAND = 10 # seconds
MED_LR_AUGMENT_VAL_NUM = -1 # Take X values left and right (X each) of median during calibration, -1 takes Q2 to Q3 range of values
STD_BOOST = 50
VOLUME_THRESH = 0.75

class Microphone:    
    # Member variables
    calibrated = False
    stream = None
    low_freq_index = 0
    high_freq_index = 0
    volume_average = 0
    avgs = [0, 0, 0, 0] # Up, Down, Left, Right, Click
    stds = [0, 0, 0, 0]

    def __init__(self):
        audio = pyaudio.PyAudio()

        # start Recording
        self.stream = audio.open(format=PYAUDIO_FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=1024)

        
    def get_main_freq(self):
        data_frames = self.stream.read(int(RATE / COLLECT_FREQUENCY), exception_on_overflow = False)
        data = np.fromstring(data_frames, NUMPY_FORMAT)
        volume_avg = np.average(abs(data)) / 1000000.0
        #volume_avg = np.var(abs(data))
        freqs = np.fft.fftfreq(data.size, d=(1.0 / RATE))
        fft_data = np.fft.fft(data)
        search_values = abs(fft_data[0:fft_data.size // 2]) # Use only half to ignore complex conjugates
        main_freq = freqs[np.argmax(search_values[self.low_freq_index:self.high_freq_index]) + self.low_freq_index]
        #print(volume_avg)
        return main_freq, volume_avg
        
    # Single run, returns True if successful calibration
    def calibrate(self, custom_print):
        """
            CALIBRATION PHASE
        """   
        data_frames = self.stream.read((int) (RATE / COLLECT_FREQUENCY), exception_on_overflow = False)
        data = np.fromstring(data_frames, NUMPY_FORMAT)
        freqs = np.fft.fftfreq(data.size, d=(1.0 / RATE))
        while(freqs[self.low_freq_index] < MIN_FREQ):
            self.low_freq_index += 1
            if (self.low_freq_index >= freqs.size):
                custom_print("Did not find specified min freq")
                return False
        self.high_freq_index = self.low_freq_index # Don't start looking from beginning
        while(freqs[self.high_freq_index] < MAX_FREQ):
            self.high_freq_index += 1
            if (self.high_freq_index >= freqs.size):
                custom_print("Did not find specified max freq")
                return False
            
        for n in range(4):
            if (n == 0):
                custom_print("Up command:")
            elif (n == 1):
                custom_print("Down command:")
            elif (n == 2):
                custom_print("Left command:")
            elif (n == 3):
                custom_print("Right command:")
            custom_print("\tHold tone for " + str(CAL_TIME_EACH_COMMAND) + " seconds...")
            
            vals = np.empty((0,))
            volume_sum = 0.0
            for i in range(int(CAL_TIME_EACH_COMMAND * COLLECT_FREQUENCY)):
                main_freq, v_a = self.get_main_freq()
                vals = np.append(vals, main_freq)
                volume_sum += v_a
                
            self.volume_average = volume_sum / (CAL_TIME_EACH_COMMAND * COLLECT_FREQUENCY)
                
            if (vals.size < 1 + MED_LR_AUGMENT_VAL_NUM * 2 or vals.size < 1):
                custom_print("\tInsuffient samples collected")
                return False
            else:
                # Try to throw out outliers
                vals_sorted = np.sort(vals)
                print(vals_sorted)
                if (MED_LR_AUGMENT_VAL_NUM < 0): # Special commands
                    vals = vals_sorted[vals_sorted.size // 4: vals_sorted.size // 4 * 3]
                else:
                    vals = vals_sorted[vals_sorted.size // 2 - MED_LR_AUGMENT_VAL_NUM: vals_sorted.size // 2 + MED_LR_AUGMENT_VAL_NUM]
                print(vals)
                
                # List statistics
                self.avgs[n] = np.average(vals)
                self.stds[n] = np.std(vals)
                custom_print("\tAverage: " + str(self.avgs[n]))
                custom_print("\tStd Dev: " + str(self.stds[n]))                    

        custom_print("Threshold adjustment...")
        #print("Before:")
        #print(self.avgs)
        #print(self.stds)

        for i in range(4):
            self.stds[i] = (self.stds[i] + 1) * STD_BOOST

        # Adjust thresholds so there is no overlap
        for i in range(4):
            for j in range(4):
                if (i != j): # Don't compare against self
                    if ((self.avgs[i] < self.avgs[j]) and ((self.avgs[i] + self.stds[i]) > (self.avgs[j] - self.stds[j]))):
                        new_std = np.abs(self.avgs[j] - self.avgs[i]) / 2
                        if (new_std < self.stds[i]):
                            self.stds[i] = new_std
                        if (new_std < self.stds[j]):
                            self.stds[j] = new_std
                            
        #print("After:")
        #print(self.avgs)
        #print(self.stds)
        
        custom_print("Finished calibration")
        self.calibrated = True
        
        return True
        
    # Continuously run    
    def get_command(self):
        """
            RUN PHASE
        """
        
        if (not self.calibrated):
            data_frames = self.stream.read((int) (RATE / COLLECT_FREQUENCY), exception_on_overflow = False)
            data = np.fromstring(data_frames, NUMPY_FORMAT)
            freqs = np.fft.fftfreq(data.size, d=(1.0 / RATE))
            while(freqs[self.low_freq_index] < MIN_FREQ):
                self.low_freq_index += 1
                if (self.low_freq_index >= freqs.size):
                    custom_print("Did not find specified min freq")
                    return -1
            self.high_freq_index = self.low_freq_index # Don't start looking from beginning
            while(freqs[self.high_freq_index] < MAX_FREQ):
                self.high_freq_index += 1
                if (self.high_freq_index >= freqs.size):
                    custom_print("Did not find specified max freq")
                    return -1
            
            self.avgs[0] = 125.0
            self.stds[0] = 50.0
            self.avgs[1] = 175.0
            self.stds[1] = 50.0
            self.avgs[2] = 225.0
            self.stds[2] = 50.0
            self.avgs[3] = 275.0
            self.stds[3] = 50.0
        
        sum = 0
        for i in range(VALUES_TO_AVERAGE):
            main_freq, v_a = self.get_main_freq()
            if v_a >= (self.volume_average * VOLUME_THRESH):
                sum += main_freq
        avg_freq = sum / float(VALUES_TO_AVERAGE)
                
        if (avg_freq > (self.avgs[0] - self.stds[0]) and avg_freq < (self.avgs[0] + self.stds[0])):
            return 'u'
        elif (avg_freq > (self.avgs[1] - self.stds[1]) and avg_freq < (self.avgs[1] + self.stds[1])):
            return 'd'
        elif (avg_freq > (self.avgs[2] - self.stds[2]) and avg_freq < (self.avgs[2] + self.stds[2])):
            return 'l'
        elif (avg_freq > (self.avgs[3] - self.stds[3]) and avg_freq < (self.avgs[3] + self.stds[3])):
            return 'r'
        else:
            return 0
          
    def close(self):
        # stop recording
        self.stream.stop_self.stream()
        self.stream.close()
        audio.terminate()