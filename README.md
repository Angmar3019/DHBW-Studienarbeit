# Entwicklung eines Prototyps zur Münzzählung mithilfe von Bildverarbeitung und Machine Learning
## Development of a prototype for coin counting using image processing and machine learning 

### 1. Meet requirements
`python<=3.9` is the main requirement for installing the following packages. I have created and tested the following program with `pyhton 3.7.3`.

Since the packages for Google Coral primarily require older libraries, Google provides detailed [requirements](https://coral.ai/software/#debian-packages) for the Debain packages.

#### Recomended
Google offers a ready-to-use image, called [aiy-maker-kit](https://github.com/google-coral/aiy-maker-kit-tools/releases/download/v20220518/aiy-maker-kit-2022-05-18.img.xz), for the Raspberry Pi suitable for use with Coral devices. All required packages and dependencies are available there, including `pyhton 3.7.3`. I would recommend to install it, because due to new packages an installation under e.g. python3.11 is almost impossible. Here you can find more information about [AIY Maker Kit](https://aiyprojects.withgoogle.com/maker/).

### 2. Install dependencies
`pip install -r req.txt`

### 3. Using the program
`python3 main.py (-ht / -t)`

`-ht` to recognize the coins by using hough transformation
<br>
`-t` to recognize the coins by using tensorflow