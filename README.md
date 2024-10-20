Run it: ```apt install geany python3-torch```. Open the .py in Geany.<br>
Replace ```python``` with ```python3``` in Geany's execute command. F5 to run.

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Terminal_4e4abe173a64d076364fff6df84783f0.png">
</p>

<br>
<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Configurable_7d3fe46d2709a004c1fcd83027ea7af6.png">
</p>

<br>
<br>

## Replace the given training-data (train.txt & test.txt)

train.txt & test.txt are the same but of unique items.
Replace them with anything like below.
"-" is normalized as "0.0" (less "seen" by the model)
while "@" is normalized as "1.0" (what the model really pays attention to.)
If ```longest``` is set to 900 for example,
then "-" is appended until your data string is 900 long.
It's completely safe. If your string is longer than ```longest```,
only the first ```longest``` characters of that string will be used.
Each line in train.txt & test.txt: ```label```, ```space```, ```- and @```, ```new line``` (\n.)
Each line in cognize.txt: ```- and @```, ```new line``` (\n.)

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

<br>
<br>

### The given training-data is a text version of MNIST, with labels & data combined

Extract training-data.tar.bz2.
**train.txt** contains 60,000 labeled images of handwritten digits
(one label & image per line of text.) An image becomes legible if you arrange
its 784 characters by rows of 28, as it is a 28x28 image.
**test.txt** contains 10,000 more such lines not found among the 60k.
See **visual_for_you.txt** to get a good look at each image.
