import pyaudio
import numpy as np

# Mic settings
PYAUDIO_FORMAT = pyaudio.paInt32
NUMPY_FORMAT = np.int32
CHANNELS = 1
RATE = 44100

# Paramters
COLLECT_FREQUENCY = 2.5 # in Hertz
OUTPUT_FREQUENCY = 10.0 # in Hertz, COLLECT_FREQUENCY / OUTPUT_FREQUENCY should be an integer
SILENCE_THRESH = 80 # ignore freq values below this
MAX_FREQ = 260 # ignore freq values above this
CAL_TIME_EACH_COMMAND = 15 # seconds
MED_LR_AUGMENT_VAL_NUM = -1 # Take X values left and right (X each) of median during calibration, -1 takes Q2 to Q3 range of values
STD_BOOST = 10

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=PYAUDIO_FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=1024)
print ("start recording...")
 
"""
    CALIBRATION PHASE
"""
avgs = [0, 0, 0, 0]
stds = [0, 0, 0, 0]
for n in range(4):
    if (n == 0):
        print("Up command:")
    elif (n == 1):
        print("Down command:")
    elif (n == 2):
        print("Left command:")
    elif (n == 3):
        print("Right command:")
    print("\tHold tone for " + str(CAL_TIME_EACH_COMMAND) + " seconds...")
    
    vals = np.empty((0,))
    for i in range((int) (CAL_TIME_EACH_COMMAND * COLLECT_FREQUENCY)):
        data_frames = stream.read((int) (RATE / COLLECT_FREQUENCY), exception_on_overflow = False)
        data = np.fromstring(data_frames, NUMPY_FORMAT)
        freqs = np.fft.fftfreq(data.size, d=(1.0 / RATE))
        fft_data = np.fft.fft(data)
        search_values = abs(fft_data[0:fft_data.size / 2]) # Use only half to ignore complex conjugates
        highest_freq_index = 0
        while(freqs[highest_freq_index] < MAX_FREQ):
            highest_freq_index += 1
            if (highest_freq_index >= freqs.size):
                print("Did not find specified max freq")
                exit();
        main_freq = freqs[np.argmax(search_values[0:highest_freq_index])]
        if (main_freq >= SILENCE_THRESH):
            #print(main_freq)
            vals = np.append(vals, main_freq)
    if (vals.size < 1 + MED_LR_AUGMENT_VAL_NUM * 2 or vals.size < 1):
        print("\tInsuffient samples collected")
        exit()
    else:
        # Try to throw out outliers
        vals_sorted = np.sort(vals)
        print(vals_sorted)
        if (MED_LR_AUGMENT_VAL_NUM < 0): # Special commands
            vals = vals_sorted[vals_sorted.size / 4: vals_sorted.size / 4 * 3]
        else:
            vals = vals_sorted[vals_sorted.size / 2 - MED_LR_AUGMENT_VAL_NUM: vals_sorted.size / 2 + MED_LR_AUGMENT_VAL_NUM]
        print(vals)
        
        # List statistics
        avgs[n] = np.average(vals)
        stds[n] = np.std(vals)
        print("\tAverage: " + str(avgs[n]))
        print("\tStd Dev: " + str(stds[n]))                    

print("Threshold adjustment")
print("Before:")
print(avgs)
print(stds)

for i in range(4):
    stds[i] = (stds[i] + 1) * STD_BOOST

# Adjust thresholds so there is no overlap
for i in range(4):
    for j in range(4):
        if (i != j): # Don't compare against self
            if ((avgs[i] < avgs[j]) and ((avgs[i] + stds[i]) > (avgs[j] - stds[j]))):
                new_std = np.abs(avgs[j] - avgs[i]) / 2
                if (new_std < stds[i]):
                    stds[i] = new_std
                if (new_std < stds[j]):
                    stds[j] = new_std
                    
print("After:")
print(avgs)
print(stds)
"""
    RUN PHASE
"""
sum = 0
counter = 1
reset_count = COLLECT_FREQUENCY / OUTPUT_FREQUENCY
if (reset_count < 1):
        reset_count = 1;
while True:
    data_frames = stream.read((int) (RATE / COLLECT_FREQUENCY), exception_on_overflow = False)
    data = np.fromstring(data_frames, NUMPY_FORMAT)
    freqs = np.fft.fftfreq(data.size, d=(1.0 / RATE))
    fft_data = np.fft.fft(data)
    search_values = abs(fft_data[0:fft_data.size / 2]) # Use only half to ignore complex conjugates
    highest_freq_index = 0
    while(freqs[highest_freq_index] < MAX_FREQ):
        highest_freq_index += 1
        if (highest_freq_index >= freqs.size):
            print("Did not find specified max freq")
            exit();
    main_freq = freqs[np.argmax(search_values[0:highest_freq_index])]
    #print(main_freq)
    if (main_freq >= SILENCE_THRESH):
        # print(main_freq)
        sum += main_freq
        if (counter < reset_count):
            counter += 1
        else:
            avg_freq = sum / reset_count
            if (avg_freq > (avgs[0] - stds[0]) and avg_freq < (avgs[0] + stds[0])):
                print("Up")
            elif (avg_freq > (avgs[1] - stds[1]) and avg_freq < (avgs[1] + stds[1])):
                print("Down")
            elif (avg_freq > (avgs[2] - stds[2]) and avg_freq < (avgs[2] + stds[2])):
                print("Left")
            elif (avg_freq > (avgs[3] - stds[3]) and avg_freq < (avgs[3] + stds[3])):
                print("Right")
            counter = 1
            sum = 0
    #msg = ""
    #for i in range((int)(np.average(np.abs(data)) / 20000000)):
    #    msg += "*"
    #print(msg)
    
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
