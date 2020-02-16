# SpecAugment [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
This is a implementation of SpecAugment that speech data augmentation method which directly process the spectrogram with Tensorflow & Pytorch, introduced by Google Brain[1]. This is currently under the Apache 2.0, Please feel free to use for your project. Enjoy!

## How to use

First, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

Next, you need to install some audio libraries work properly. To install the requirement packages. Run the following command:

```bash
pip3 install SpecAugment
```

And then, run the specAugment.py program. It modifies the spectrogram by warping it in the time direction, masking blocks of consecutive frequency channels, and masking blocks of utterances in time.

#### *Try your audio file SpecAugment*

```shell
$ python3
```

```python
import librosa
from specAugment import spec_augment_tensorflow
# If you are Pytorch, then import spec_augment_pytorch instead of spec_augment_tensorflow
audio, sampling_rate = librosa.load(audio_path)
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)


# Visualize
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max))
plt.show()

# Do SpecAugment
mel_spectrogram = torch.Tensor(mel_spectrogram).unsqueeze(0)
warped_masked_spectrogram = spec_augment(mel_spectrogram=mel_spectrogram).squeeze().numpy()

# Visualize
librosa.display.specshow(librosa.power_to_db(warped_masked_spectrogram, ref=np.max))
plt.show()
```
Learn more examples about how to do specific tasks in SpecAugment at the test code.

```bash
python spec_augment_test.py
```
In test code, we using one of the [LibriSpeech dataset](http://www.openslr.org/12/).

<p align="center">
  <img src="https://github.com/shelling203/SpecAugment/blob/master/images/Figure_1.png" alt="Example result of base spectrogram"/ width=600>
  <img src="https://github.com/shelling203/SpecAugment/blob/master/images/Figure_2.png" alt="Example result of base spectrogram"/ width=600>
</p>


# Reference

1. https://arxiv.org/pdf/1904.08779.pdf
