# Speech enhancement demo

Demo application for speech denoising algorithm.

## Running

DSEGAN:

`
python "speech_enhancement_demo/speech_enhancement_demo.py" -at dsegan -m "D:/DSEGAN-2/SEGAN-97000.meta" -i "data/noisy_signal.wav" -o "data/clean_signal_dsegan.wav" -depth 2
`

ISEGAN:

`
python "speech_enhancement_demo/speech_enhancement_demo.py" -at isegan -m "D:/DSEGAN-2/SEGAN-97000.meta" -i "data/noisy_signal.wav" -o "data/clean_signal_isegan.wav" -iter 2
`

SEGAN:

`
python "speech_enhancement_demo/speech_enhancement_demo.py" -at segan -m "D:/DSEGAN-2/SEGAN-97000.meta" -i "data/noisy_signal.wav" -o "data/clean_signal_segan.wav"
`

## Arguments description

| key |type |description | required |
|-----|---|---|---|
| -h  |  | Help | |
| -i  | str | Noisy audio file to process. | true |
| -m  | str | Path to a .meta file with a trained model. | true |
| -at | str | Type of the network, either 'dsegan' for deep SEGAN, 'isegan' for iterated SEGAN or 'segan' for SEGAN. | true |
| -o  | str | The output file in .wav format, where the clean audio file will be stored. | true |
| -depth  | int | The depth of DSEGAN (default=1). | false |
| -iter  | int | The number of iterations of ISEGAN (default=1).| false |
| -p   | float | The preemph coeff (default=0.95). | false |
| -d   | str | Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. The sample will look for a suitable plugin for device specified (default=CPU). | false |


## About

The demo produces a noise-free audio recording as an output file. 


**Preprocessing:**

Isegan, dsegan and segan models accept audio recording in .wav format with a sampling rate of 16kHz. Our demo accepts audio recording at any sample rate as input. File resampling up to 16kHz is performed using librosa package tools. The librosa.load function loads the audio file at the required sampling rate and performs normalization. Therefore, we removed a part of the code with normalization as unnecessary.

**Postprocessing:**

We have noticed that the output demo file is cleared of noise, but is very quiet, which made it difficult to assess the quality of the cleansing. In the postprocessing part, the audio volume is increased using the python package Pydub (https://github.com/jiaaro/pydub).




