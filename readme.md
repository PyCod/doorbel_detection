# Simple raspberry pi bell detector

To recreate for your own bell do the following:

1. Get a raspberry pi and a microphone in a spot of your choice, where it has internet access and can hear the bell ringing. Do the steps in `extra_deps.txt`

  Tip: A shittier mic is better in this case, as it does not filter out background noise, aka the bell, that we want to hear

2. Record a sample of normal noise in the house (mine was 1h long, but does not need to be that long, I also used arecord for the recording), put in raw_data folder
3. Record several bell rings of differing lengths in one recording, put in raw data folder
4. Extract the bell sounds with a tool like audacity, cutting out small fragments longer than 0.5s which have all bell noise inside, put them in the split_data folder as bell_1.wav, bell_2.wav ...
5. Follow notebook order in notebooks folder, adjusting code to your needs
6. You should now have a trained model, copy the 3 model files (<model_name>, <model_name>.arff and <model_name>MEANS) to `bell_detector/model`
7. Adjust model name in code of main.py, also adjust your pushover account info
8. Copy whole `bell_detector` folder to the Rpi, create a virtual environment and install requirements
9. Run main.py in something like screen or nohup
10. Profit, enjoy your detector!