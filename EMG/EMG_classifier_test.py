from EMG_classification import EMG_Classifier
import numpy as np

def main():

	zeros_test = np.zeros(950)
	emgc = EMG_Classifier(True)
	print(emgc.classify(zeros_test))

if __name__ == "__main__":
  main()