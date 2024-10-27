Run it: ```apt install geany python3-torch```. Open the .py in Geany.<br>
Replace ```python``` with ```python3``` in Geany's execute command. F5 to run.

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Terminal_4e4abe173a64d076364fff6df84783f0.png">
</p>

<br>
<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/compromise-evident/ML/refs/heads/main/Other/Configurable_a9b59da0471de8ddb833fb4ac3b6bae5.png">
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

## When you're done playing with weak configurations

* Set ```depth``` to ~5
* Set ```width``` to ~5000
