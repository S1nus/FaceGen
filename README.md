# FaceGen
![Ghost in the Machine](ghostinmachine.png)
* FaceGen is a Generative Adversarial Neural Network (GAN) which dreams of human faces.
* It works by importing 10,000 .jpg files of photos of people's faces, vectorizing them, then training two neural networks (a generative, and a discriminator) on that data in order to work against each other to converge on a brand new face.

# Scalability
* At the moment, only FacesSmall.py works, which is only capable of drawing faces in 28x28 pixel images.
* Faces.py is a 128x128 pixel "scaled-up" version of FacesSmall, but unfortunately I haven't quite worked out all the bugs. There's currently an error relating to the algorithm used by TensorFlow, which I'm 90% finished fixing. I just need to figure out what size to make the "filter" Tensor so that the result of the 2dConvolution of the input data to the generator results in the correct sized tensor of the output.
* Also, it unfortunately is only in black and white, however, after I figure out how to scale it up to 128x128, all I would need to do to get it to work in color is to replace the 1-dimensional color array with a 3-dimensional color array (for RGB), and tweak the algorithm to include that.
