# Speech enhancement demo

Example of running:
`
python "speech_enhancement_demo/speech_enhancement_demo.py" -at segan -m "D:/DSEGAN-2/SEGAN-97000.meta" -i "data/noisy_signal.wav" -o "data/clean_signal.wav"
`

## Arguments description

| key|type |description | required |
|----|---|---|---|
| -h |  | help | |
| -i | str | Noisy audio file to process. | true |
| -m | str | Path to a .meta file with a trained model.t | true |
| -at | str | Type of the network, either 'dsegan' for deep SEGAN, 'isegan' for iterated SEGAN or 'segan' for SEGAN. | true |
| -o | str | The output file in .wav format, where the clean audio file will be stored. | true |
| -depth | int | The depth of DSEGAN. | false |
| -iter | int | The number of iterations of ISEGAN.| false |
| -p | float | The preemph coeff. | false |
| -d | str | Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. The sample will look for a suitable plugin for device specified. | false |
