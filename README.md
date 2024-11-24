Run it: ```apt install geany python3-torch```. Open the .py in Geany.<br>
Replace ```python``` with ```python3``` in Geany's execute command. F5 to run.

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Terminal_4e4abe173a64d076364fff6df84783f0.png">
</p>

<br>
<br>

## Replace the given training-data (train.txt & test.txt)

```text

   label                       your data
     |                             |


     2 ------@@@-@--------@@@@@@-------------------------@
     9 @--------------------------------------------------------@@
     8 ----------------------------@@
     8 --------------------------@@-------------------------
     0 -------@-@-@-@
     36 --------------@-@@@@-@
     300 --------------------------------------@-@
     5000 ----------@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-------@
     0 --------@-

```

The given training-data is a text version of MNIST, with labels & data combined like above.
**train.txt** contains 60,000 labeled images of handwritten digits
(one label & image per line of text.) An image becomes legible if you arrange
its 784 characters "---@@@---" by stacking rows of 28 (it's a 28x28 image.)
**test.txt** contains 10,000 more such lines not found among the 60k.
See **visual_for_you.txt** to see each image
already stacked 28x28.

So, train.txt & test.txt are the same but of unique items.
Replace them with anything like above.
"-" is normalized as "0.0" (less "seen" by the model)
while "@" is normalized as "1.0" (what the model pays attention to.)
If ```longest``` is set to 900 for example,
then "-" is appended until your data string is 900 long.
It's completely safe. If your string is longer than ```longest```,
only the first ```longest``` characters of that string will be used.
Each line in train.txt & test.txt: ```label```, ```space```, ```- and @```, ```new line``` (\n.)
Each line in cognize.txt: ```- and @```, ```new line``` (\n.)
(You have to create cognize.txt.)

<br>
<br>

--------------------------------------------------------

--------------------------------------------------------

--------------------------------------------------------

--------------------------------------------------------

--------------------------------------------------------

<br>
<br>

## (WIP) GPU - follow these simplified steps in order

```text
SSD    $ 43 amazon.com/dp/B09ZYPTXS4
PSU    $ 45 amazon.com/dp/B0B4MVDRX4
RAM*2  $ 80 amazon.com/dp/B0143UM4TC
MB     $ 90 amazon.com/dp/B08KH12V25 (comes with 2 SATA data cables)
CPU    $160 amazon.com/dp/B091J3NYVF
GPU    $285 amazon.com/dp/B08WPRMVWB
```
* Connect the PSU 24-pin, and 8-pin (4+4) connectors to the bare motherboard.
* Format a thumb drive to fat32.
* Go to ASUS support for that motherboard, download the latest BIOS update, extract it.
* Rename the .CAP file to "PB450MA2.CAP" and copy it to the thumb drive.
* Insert the thumb drive into the motherboard USB port closest to the HDMI port.
* Press and hold the motherboard's Flashback button for 3 seconds.
* In a few minutes, the motherboard will stop blinking.
* Assemble (motherboard takes PSU 24-pin and 8-pin (4+4) connectors. GPU takes PSU 8-pin (6+2) connector.
* If only 2 RAM sticks, insert them into only the grey slots, or only the black slots on the motherboard.
* Safe preset overclock: change "Normal" to "Optimal" in BIOS, and update BIOS time (24-hour time.)
* Do a fresh install of Devuan Linux (or Debian-based Linux).
* ```apt install firmware-amd-graphics xserver-xorg-video-amdgpu``` then reboot. (Full res for CPU's iGPU.)
* ```apt install nvidia-driver``` then reboot.
* ```nvidia-smi``` (just to verify) - should show driver version & GPU details.
* ```apt install nvidia-cuda-toolkit```.
* ```nvcc --version``` (just to verify) - should show CUDA version.
* ```apt install python3-torch```.
* ```apt install geany psensor```.
* Open ML.py in Geany (the text editor you just installed.)
* Replace "python" with "python3" in Geany's execute command.
* Replace 'cpu' with 'cuda' in ML.py. F5 to run.
* Open psensor (the temperature display you just installed.)
* Finally, put a momentary-switch on the motherboard power button pins below.

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Power_button_pins.png">
</p>
