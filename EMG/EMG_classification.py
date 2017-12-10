from time import sleep
import numpy as np
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from matplotlib.pyplot import draw, figure, show
import sys
from moving_average import moving_avg_filter_t

"""

Left, under eye: Channel 0
Left, above eye: Channel 1
Right, above eye: Channel 2
Right, under eye: Channel 3

"""
 # 10 instances; right, under eye
TRAINING_SAMPLE_FILE_1 = "recorded_samples/scenario1.txt"
TRAINING_SAMPLE_FILE_2 = "recorded_samples/scenario2.txt"
TRAINING_SAMPLE_FILE_3 = "recorded_samples/scenario3.txt"
TRAINING_SAMPLE_FILE_4 = "recorded_samples/scenario4.txt"
TRAINING_SAMPLE_FILE_5 = "recorded_samples/scenario5.txt"
TRAINING_SAMPLE_FILE_6 = "recorded_samples/scenario6.txt"
TRAINING_SAMPLE_FILE_7 = "recorded_samples/scenario7.txt"
TRAINING_SAMPLE_FILE_8 = "recorded_samples/scenario8.txt"
TRAINING_SAMPLE_FILE_9 = "recorded_samples/scenario9.txt"
TRAINING_SAMPLE_FILE_10 = "recorded_samples/scenario10.txt"

#make a vector for all scenario files
files = [TRAINING_SAMPLE_FILE_1, TRAINING_SAMPLE_FILE_2, TRAINING_SAMPLE_FILE_3, TRAINING_SAMPLE_FILE_4, TRAINING_SAMPLE_FILE_5, TRAINING_SAMPLE_FILE_6, TRAINING_SAMPLE_FILE_7, TRAINING_SAMPLE_FILE_8, TRAINING_SAMPLE_FILE_9, TRAINING_SAMPLE_FILE_10]

# Parameters
EMG_SAMPLING_RATE = 1000
FFT_SIZE = 200
POSITIVE = 1
NEGATIVE = 0
SAMPLE_SIZE = 1000
NUM_CLUSTERS = 10
TRAINING_PORTION = 0.7
REDUCED_BINS_SIZE = 200
TEST_LOOP_TIMES = 20

BIN_TESTS = [1, 5, 10, 20, 25, 50, 100, 200, 250, 500]

fileOfBlinks = open("training.csv")
row = []
csvFile = []

class EMG_Classifier:

    def __init__(self):
        
        master_file = []
        ch_filter = moving_avg_filter_t(250, 250, -0.25, 500)
        
        #step 1
        for file in files:
            print("Parsing " + file)
            
            training_samples = dict()
            training_samples['ch0'] = []
            training_samples['ch1'] = []
            training_samples['ch2'] = []
            training_samples['ch3'] = []
        
            f_obj = open(file)
            # Skip first line
            for line in f_obj.readlines()[1:]:
                line = line.strip()
                [ch, values_str] = line.split(' :')
                values = values_str.split('[')[1].split(']')[0].split(', ')
                int_array = [int(numeric_string) for numeric_string in values]
                training_samples[ch].extend(int_array)
            
            channels = ['ch0', 'ch1', 'ch2', 'ch3']
            for ch in channels:
                # Demean
                filtered_samples = []
                for sample in training_samples[ch]:
                    [diff_sample, fast_op, slow_op, detection_value] = ch_filter.signal_detect(sample)
                    filtered_samples.append(slow_op)
                training_samples[ch] = filtered_samples
                #pyplot.plot(filtered_samples)
                #pyplot.show()
            master_file.append(training_samples)
    
        #step 2
        #parse csv file
        for line in fileOfBlinks.readlines():
            [file, channel, startTime] = line.split(',')
            csvFile.append([int(file), 'ch' + channel, int(1000*float(startTime))])
        
        for bin_size in BIN_TESTS:
            segmented_data = []
            segmented_labels = []

            prev_s = -1
            prev_c = -1
            prev_seg_end = 0
            for line in range(len(csvFile)):
                #print(line)
                # New section
                s = csvFile[line][0] - 1 # Scenarios start are 1, but the arrays are 0-indexed
                c = csvFile[line][1]
                sp = csvFile[line][2] # starting point
                #print("Debug: " + str(s + 1) + ", " + str(c) + ", " + str(sp))
                #pyplot.plot(master_file[s][c])
                #pyplot.show()
                if (s != prev_s or c != prev_c):
                    prev_seg_end = 0
                    # Take all prev neg examples if available
                    if (prev_s > 0 and prev_c > 0):
                        negative_loops = int((sp - prev_seg_end) / SAMPLE_SIZE)
                        for chunk in range(negative_loops):
                            #print("Negative prev")
                            seg_start = prev_seg_end
                            prev_seg_end = seg_start + SAMPLE_SIZE
                            #print(seg_start)
                            #print(prev_seg_end)
                            target_segment = master_file[prev_s][prev_c][seg_start:prev_seg_end]
                            #print(len(target_segment))
                            segmented_data.append(self.segment_processing(target_segment, False, bin_size))
                            segmented_labels.append(NEGATIVE)
                        
                if (prev_seg_end > sp):
                    # Error in training file, abort
                    print("Error: " + str(s + 1) + ", " + str(c) + ", " + str(sp))
                    exit()

                # Get all negative samples before starting point
                negative_loops = int((sp - prev_seg_end) / SAMPLE_SIZE)
                #print(negative_loops)
                for chunk in range(negative_loops):
                    #print("Negative")
                    seg_start = prev_seg_end
                    prev_seg_end = seg_start + SAMPLE_SIZE
                    #print(seg_start)
                    #print(prev_seg_end)
                    target_segment = master_file[s][c][seg_start:prev_seg_end]
                    #print(len(target_segment))
                    segmented_data.append(self.segment_processing(target_segment, False, bin_size))
                    segmented_labels.append(NEGATIVE)

                # Get 1 positive sample if possible
                if (sp + SAMPLE_SIZE < len(master_file[s][c])):
                    #print("Positive")
                    prev_seg_end = sp + SAMPLE_SIZE
                    target_segment = master_file[s][c][sp:prev_seg_end]
                    segmented_data.append(self.segment_processing(target_segment, True, bin_size))
                    segmented_labels.append(POSITIVE)

            #print (segmented_labels)
            
            #print("Kmeans: ")
            kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(segmented_data)
            #print(kmeans.labels_)
            #print(segmented_labels)
            # Find purity
            for i in range(NUM_CLUSTERS):
                #print(np.where(kmeans.labels_ == i)[0])
                grouped_labels = np.array(segmented_labels)[np.where(kmeans.labels_ == i)[0]]
                counts = np.bincount(grouped_labels)
                #print("Group " + str(i) + " marked as " + str(np.argmax(counts)) + ", size: " + str(grouped_labels.size) + ", purity: " + str(counts[np.argmax(counts)] / float(grouped_labels.size)))
            
            # RBF Kernal
            guass_rbf = GaussianProcessClassifier(1.0 * RBF(1.0))
            print("RBF: ")
            avg = 0
            for i in range(TEST_LOOP_TIMES):
                random_indicies = np.arange(len(segmented_data))
                np.random.shuffle(random_indicies)
                training_indicies = random_indicies[:int(random_indicies.size * TRAINING_PORTION)]
                testing_indicies = random_indicies[int(random_indicies.size * TRAINING_PORTION):]
                guass_rbf.fit(np.array(segmented_data)[training_indicies], np.array(segmented_labels)[training_indicies])
                predicted_labels = guass_rbf.predict(np.array(segmented_data)[testing_indicies])
                #print("RBF accuracy: " + str(accuracy_score(np.array(segmented_labels)[testing_indicies], predicted_labels)))
                tn, fp, fn, tp = confusion_matrix(np.array(segmented_labels)[testing_indicies], predicted_labels).ravel()
                #print("RBF TN/FP/FN/TP: " + str(tn) + "/" + str(fp) + "/" + str(fn) + "/" + str(tp))
                #print(tp/(fn + tp))
                avg += tp/(fn + tp)
            avg /= TEST_LOOP_TIMES
            print(str(bin_size) + " average: " + str(avg))
         
        ch_name = "ch0"
        detected_values = []
        ch_filter = moving_avg_filter_t(250, 250, -0.25, 500)

        f1 = figure()
        af1 = f1.add_subplot(111)
        f2 = figure()
        af2 = f2.add_subplot(111)
        af2.plot(training_samples[ch_name])
        show(block=False) 

        while sample_index < len(training_samples[ch_name]):
            samples = training_samples[ch_name][sample_index:sample_index+FFT_SIZE]
            #filtered_samples = []
            #for sample in samples:
            #    [diff_sample, fast_op, slow_op, detection_value] = ch_filter.signal_detect(sample)
            #    filtered_samples.append(slow_op)
            #samples = filtered_samples
            af1.plot(samples)
            pyplot.xlabel("Sample index = {0}".format(sample_index))
            pyplot.show()
        
            # Demean the channel
            demeaned_channel = samples - np.mean(samples)
        
            freqs = np.fft.fftfreq(len(demeaned_channel), d = 1.0 / (EMG_SAMPLING_RATE))
            fft_powers = abs(np.fft.fft(demeaned_channel))
        
            af1.plot(freqs[:int(freqs.size / 4)], fft_powers[:int(freqs.size / 4)])
            
            pyplot.xlabel("Sample index = {0}".format(sample_index))
            pyplot.show()

            sm = np.sum(fft_powers[0:5])/np.sum(fft_powers[0:10])
            if sm > 0.73:
                detected_values.append(1)
            else:
                detected_values.append(0)
            print("Mean for {0:4d} - {1:4d} => {2:.2f}".format(sample_index, sample_index + FFT_SIZE, sm))
            sample_index += FFT_SIZE
        af1.plot(detected_values)
        show()
        while True:
            sleep(1)
 
    def segment_processing(self, segment, pos, bin_size):
        segment -= np.mean(segment)
        
        """
        pyplot.plot(segment)
        if (pos):
            pyplot.title("Positive")
        else:
            pyplot.title("Negative")
        pyplot.show()
        """
        
        """ FFT into bins of lower frequencies
        #fft_powers = abs(np.fft.fft(segment))
        #processed = fft_powers[:int(fft_powers.size / 4)]
        """
        
        #""" Scaled time domain
        seg_max = segment[np.argmax(segment)]
        seg_min = segment[np.argmin(segment)]
        pre_reduction = segment / float(seg_max - seg_min)
        # Break down bins
        processed = []
        average_amount = int(SAMPLE_SIZE / bin_size)
        #print("Before: " + str(len(pre_reduction)))
        for i in range(bin_size):
            processed.append(np.average(pre_reduction[i*average_amount:(i+1)*average_amount]))
        #print("After: " + str(len(processed)))
        #"""
        
        #pyplot.plot(processed)
        #pyplot.show()
        return processed
 
    def classify_chunk(self):
        pass
