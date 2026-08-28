<img src="./assets/deepwater_horizon_oil_spill_may_24_2010.jpg" width="1000">

# Paper List
These papers below use the same general type of hyperspectral dataset (HOSD: 1, 2, 3. Similar but different world location: 4 ). The HOSD dataset is not a simulation (eg, the researchers mix their own oil in a lab). There are a few more papers that use hyperspectral cameras but are simulated (5, 6)

*Cited with APA 7*

[1] Duan, P., Kang, X., Ghamisi, P., & Li, S. (2023). Hyperspectral remote sensing benchmark database for oil spill detection with an isolation forest-guided unsupervised detector. IEEE Transactions on Geoscience and Remote Sensing, 61, 1-11.

[2] Kang, X., Deng, B., Duan, P., Wei, X., & Li, S. (2023). Self-supervised spectral–spatial transformer network for hyperspectral oil spill mapping. IEEE Transactions on Geoscience and Remote Sensing, 61, 1-10.

[3] Samkhaniani, M., Khoshand, A., & Ezati, S. (2026). Deep learning-based hyperspectral oil spill detection for marine pollution monitoring in the Gulf of Mexico: A step toward marine pollution monitoring and SDG 14 compliance. Marine Pollution Bulletin, 222, 118908.

[4] Yang, J., Wang, J., Hu, Y., Ma, Y., Li, Z., & Zhang, J. (2023). Hyperspectral marine oil spill monitoring using a dual-branch spatial–spectral fusion model. Remote Sensing, 15(17), 4170.

[5] Carrasco-García, M. G., Rodríguez-García, M. I., González-Enrique, J., Ruiz-Aguilar, J. J., & Turias-Domínguez, I. J. (2023). Hyperspectral technology for oil spills characterisation by using feature selection. Transportation Research Procedia, 71, 117-123.

[6] Carrasco-García, M. G., Rodríguez-García, M. I., Ruíz-Aguilar, J. J., Deka, L., Elizondo, D., & Turias Domínguez, I. J. (2024). Oil spill classification using an autoencoder and hyperspectral technology. Journal of Marine Science and Engineering, 12(3), 495.


## [1] Hyperspectral Remote Sensing Benchmark Database for Oil Spill Detection with an Isolation Forest-Guided Unsupervised Detector
This paper is the source of our main HOSD dataset. They wanted to develop a way to generate accurate ground truth samples for oil spill because manual labelling requires a lot of effort. To be able to compare their labelling method, they had "field experts" label "pixel by pixel" using Environment for Visualizing Images (ENVI) software. In the end, they showed their unsupervised oil spill detection method that uses "both the spectral and spatial information of oil spill regions" to automatically generate training samples for the spectral classifier.

### Data Information
#### Data Source
Gulf of Mexico oil spill with AVIRIS sensor. NASA provided AVIRIS images. The spectral coverage is from 365nm to 2500nm. Due to different altitudes in different routes, the spatial resolution of images vary.

#### Preprocessing
There are many preprocessing levels for data in general, such as:

0 - Raw Data

1 - Radiance

1A - Radiance

1B - Radiance, Sensor Coordinates

2 - Geophys. Variables, Sensor Coordinates

3 - Gridded Observations

4 - Gridded Model Output

5 - Non of the above

The HOSD dataset aligns most with *2 - Geophys. Variables* because it was converted to surface reflectance before the authors applied their algorithm: "These datasets have been processed with an atmospheric correction model in ENVI 5.3 software before oil spill detection, where the Fast Line-of-Sight Atmospheric Analysis of Spectral Hypercube (FLAASH) is adopted."

#### Training Method
Noisy Band Removal: "A Gaussian statistical-based method is developed to automatically remove the bands corrupted by serious noise."

Dimensionality Reduction: "The kernel principal component analysis (KPCA) is employed to reduce the high dimensionality of the HSIs."

Probability Estimation: "The isolation forest is employed to detect the probability of each pixel belonging to the oil film and the seawater..."

Pseudo-Label Generation: "...an efficient k-means algorithm is performed on the probability map $p$ to generate the training samples."

Initial Classification: "1% of the whole training samples are randomly selected... The training set is fed into the SVM classifier to yield the initial detection map."

Spatial Optimization: "The ERW [extended random walker] algorithm is utilized to optimize the initial detection map by integrating the spatial information so as to produce the final result."

#### Recommended Detection Metrics
The paper suggests these two main metrics that are good for anomaly detection.

ROC Curve: The probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance. 

Precision: True positive rate over all positive predictions.


#### Results
"The satisfactory detection results obtained by the proposed method are mainly due to several reasons: First, the noisy band removal can effectively alleviate the interference of severe noisy bands to detection performance. This step is able to greatly boost the robustness of the proposed method. Second, the isolation forest method produces relatively true training samples, even though it does not involve human annotation. Third, the ERW-based optimization process takes full advantage of the spatial correlations among neighboring pixels. This optimization is beneficial for improving the detection accuracy."

<img src="./assets/hosdresults.png" width="500">

#### Ablations
<img src="./assets/hosdablations.png" width="500">


## [2] Self-Supervised Spectral–Spatial Transformer Network for Hyperspectral Oil Spill Mapping

## [3] Deep learning-based hyperspectral oil spill detection for marine pollution monitoring in the Gulf of Mexico: A step toward marine pollution monitoring and SDG 14 compliance

## [4] Hyperspectral Marine Oil Spill Monitoring Using a Dual-Branch Spatial–Spectral Fusion Model

## [5] Hyperspectral technology for oil spills characterisation by using feature selection

## [6] Oil Spill Classification Using an Autoencoder and Hyperspectral Technology