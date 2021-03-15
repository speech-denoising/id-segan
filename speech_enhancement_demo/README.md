# Speech enhancement demo

Example of running:
`
python "speech_enhancement_demo/speech_enhancement_demo.py" -at segan -m "D:/DSEGAN-2/SEGAN-97000.meta" -i "data/noisy_signal.wav" -o "data/clean_signal.wav"
`

## Arguments description

| key|type |description |
|----|---|---|
| -h |  | help |
| -i | str | Noisy audio file to process. |
| -m | str | Path to a .meta file with a trained model.t |
| -at | str | Type of the network, either 'dsegan' for deep SEGAN, 'isegan' for iterated SEGAN or 'segan' for SEGAN. |
| -o | str |  |
| -depth | int |  |
| -iter | int | |
| -p | float |  |
| -d | str |  |
