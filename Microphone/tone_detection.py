from Microphone import Microphone
    
def main():
    mic = Microphone()
    
    if (not mic.calibrate(print)):
        exit()
    
    while True:
        print(mic.get_command())
        
    mic.close()

if __name__ == "__main__":
    main()
