# Experimentation process
I initially tried convolutional with max pooling and flattening and drop out --> got 5% accuracy

I then thought, maybe since its 30x30 already, we shouldn't max pool?
I tried this on the small data set and got like 99%, but maybe that's just because it is small so I tested it on the medium dataset.
I got 13%

Switched from one-hot to sparse-categorical because having a bunch of 1s seems dumb.
Apparently that changes the output size and breaks it
Also apparently the pooling is important

The biggest fix was noramlizing pixels. It reduces the range but keeps the information there. Larger inputs cause larger gradients. Also, prevents large values from having too much weight. (94% accuracy on medium set)

I added another conv2d layer (learned about the output shape, you only need input shape on first layer, increase filters as you go through layers)
I realized the 30x30x3 is 3 for rbg colors, and turns into 32 for 32 filters. Max pooling 2x2 halves the inputs. (96.7% accuracy)

Data transformation - rotation, shifts, zoom (87% accuracy)
I guess these random transformations add very little value.
They actually confuse the model for some reason.
I moved epochs from 10 to 15, (90% accuracy).
