# SCC0251_ImageProcessing_FinalProject
Final project for the course SCC0251 Image Processing offered by Institute of Mathematical Sciences and Computation from University of São Paulo in 2022.

## Project: Extracting rivers from satellite images
This project aims to test image processing methods, such as tresholding, mathematical morphology and Canny edges detection, to extract rivers' shapes from satellite images. We aim to compare our results among different rivers captures, working with different colour channels and other image treating methods according to each specific case in order to find whether there are a most advantageous algorithm or method for extraction for each one.
We plan to use images avaiable for public access by INPE (www.dg.inpe.br) as input.

So far, we tested two different approaches for river detection: **threshold** and **edge-detection** with Canny's algorithm, which some results are presented below. The code development and discussion for each one is done in their respective notebooks in the "codes" directory.

### Thresholding

![Results](/readme_images/results2.png?raw=true "Thresholding results")

### Canny edge extration algorithm

The three different colour channels for a river image (imagens/clean/amaz20.jpg).
![Results](/readme_images/canny_channels_before.png?raw=true "Image different channels before")

The results for each of them after applying Canny's algorithm.
![Results](/readme_images/canny_channels_after.png?raw=true "Image different channels after edge detection")
