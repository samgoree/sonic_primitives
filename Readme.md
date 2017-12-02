# Sonic Primitives

An adaptation of the ideas behind [Primitive.lol](https://github.com/fogleman/primitive) for sounds. See [my blog](https://samgoree.github.io/2017/12/01/Sonic-Primitives.html) for details.

### Dependencies

The program is written in Python 3 and depends on [Numpy](http://www.numpy.org/).

### Usage

`python sonic_primitives.py <wav file> <output file prefix>`

Where `<wav file>` encodes the sound you want to approximate and <output file prefix> is the path where you want to output files, which will be appended with the iteration number and `.wav` for each saved iteration (currently saves every 20 new waves).

If this repo gets any attention, I will migrate to argparse and add all of the hyperparameters as arguments, but for now, hyperparameters are set as constants at the top of the source code.