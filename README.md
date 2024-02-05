# Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
## Development of a prototype for coin counting using image processing and machine learning 

### 1. Meet requirements
`python<=3.9` is the main requirement for installing the following packages. I have created and tested the following program with `pyhton 3.9`.

Since the packages for Google Coral primarily require older libraries, Google provides detailed [requirements](https://coral.ai/software/#debian-packages) for the Debain packages.

#### Recomended:
Google offers a ready-to-use image, called [aiy-maker-kit](https://github.com/google-coral/aiy-maker-kit-tools/releases/download/v20220518/aiy-maker-kit-2022-05-18.img.xz), for the Raspberry Pi suitable for use with Coral devices. All required packages and dependencies are available there, including `pyhton 3.7.3`. I would recommend to install it, because due to new packages an installation under e.g. python3.11 is almost impossible. Here you can find more information about [AIY Maker Kit](https://aiyprojects.withgoogle.com/maker/).

1. Change to bulleseye: <br>
    `/etc/apt/sources.list`<br>
    `/etc/apt/sources.list.d/raspi.list`

#### Pi camera V3:
If you want to use a Pi camera V3, you need at least debian bullseye, because you need the `libcamera` package. The aiy-maker-kit image contains debian buster, which is too old for it. You can use the image as I did and then simply upgrade to bulleseye, which uses `python 3.9`. Here is a [guide](https://www.tomshardware.com/how-to/upgrade-raspberry-pi-os-to-bullseye-from-buster) for upgrading.

1. Update your raspberry pi with `rpi-update`
2. Disable `camera interface` in `raspi-config`
3. Change following parameters in `/boot/config.txt` to:<br>
    `dtoverlay=vc4-kms-v3d`<br>
    `dtoverlay=imx708`<br>
    `camera_auto_detect=0`<br>

### 2. Install dependencies
`pip install -r req.txt`

`sudo apt install python3-opencv python3-numpy python3-picamera2 libcamera-apps`

### 3. Using the program
`python3 main.py (-ht / -t and -l)`

`-ht` to recognize the coins by using hough transformation <br>
`-t` to recognize the coins by using tensorflow<br>

`-t` {and path to model}<br>
`-l` {and path to labels}

Example:
`pyhton main.py -t model_quant_edgetpu.tflite -l labels.txt`

### 4. Training your own model
If you would also like to train your own model, based on your own images and parameters, you can use the following code and settings.

#### Label Studio
I recommend [Label Studio](https://github.com/HumanSignal/label-studiolabelstudio) for labeling the images. This allows you to label the images and export them as coco format. The following code for training is written specifically for use with the coco format and Label Studio. If you want to use other software, the following code may not work.

To get started, follow the Label Studio [instructions](https://github.com/HumanSignal/label-studio?tab=readme-ov-file#try-out-label-studio) to install and create a project. Then you can import your images or those provided by me. Under `Projects > Train > Settings > Labeling Interface` please insert the following template, which provides the categories for labeling.

```
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="1" background="#006400"/>
    <Label value="2" background="#ff0000"/>
    <Label value="5" background="#ffd700"/>
    <Label value="10" background="#c71585"/>
    <Label value="20" background="#00ff00"/>
    <Label value="50" background="#00ffff"/>
    <Label value="100" background="#0000ff"/>
    <Label value="200" background="#1e90ff"/>
  </RectangleLabels>
</View>
```

#### Google Collab


I would recommend using [Google Collab](https://colab.research.google.com/) for the training process. Google Collab provides a jupyter notebook environment where you can also add an Nvidia T4 GPU. This speeds up the training process considerably and can run in the background.

Simply open the `model_training.ipynb` in Google Collab and follow the instructions given by the notebook. The notebook is specially adapted to Google Collab and its functionality, if you want to use it locally you have to customize it