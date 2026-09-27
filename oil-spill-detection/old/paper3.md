# The Potential of Hyperspectral Image Classification for Oil Spill Mapping
Xudong Kang; Zihao Wang; Puhong Duan; Xiaohui Wei

## I. Introduction

With the rapid development of marine oil transportation and offshore oil exploration, oil spill accidents occur more frequently than ever before, which poses a serious threat to the ecology, environment, and so on [1], [2], [3], [4], [5], [6]. For instance, in 1978, the Amoco Cadiz ran aground in a storm off Brittany, France, leaking about 250,000 ton of light crude oil [7]. During the 1991 Gulf War, Iraqi forces opened a valve that spilled 56,000 ton of oil into the Gulf of Mexico [8], [9]. In 2018, the “SANCHI” oil tanker collided with the bulk carrier in the East China Sea, and 130,000 ton of condensate oil spill poured into the Yangtze River, polluting 100 km² of seawater [10], [11]. In 2021, AMGhent suffered an equipment failure while refueling in the Gulf of Gibraltar, spilling 2000 to 5000 l of fuel. These accidents caused by natural disasters, human activities, and equipment failures have seriously harmed ecological health and fishery development [12], [13], [14], [15]. Therefore, there is a growing demand for monitoring the distribution of oil film and the levels of pollution [16], [17], [18], [19], [20].

Over the past years, many imaging techniques have been applied for oil spill monitoring, including synthetic aperture radar (SAR) images [21], [22], [23], [24], [25], multispectral images (MSIs) [26], [27], [28], [29], and hyperspectral images (HSIs) [30], [31], [32], [33]. SAR images and MSIs mainly are used to detect polluted regions while failing to distinguish different types of oil films, which limits their applications. On the contrary, hyperspectral imaging is a mushroom growing technique, which can record both abundant spectral information and finer spatial information [34], [35], [36], [37]. Due to this merit, HSIs have been widely used for oil film thickness estimation and oil film type identification. Up until now, many HSI classification methods have been studied to identify different materials, such as minerals, crops, and oil films, which can be roughly divided into three categories (see Fig. 1): feature extraction, classifier, and postprocessing.

**Fig. 1. General workflow for hyperspectral oil spill mapping.**

Feature extraction methods aim to reduce the spectral dimension of original data and obtain discriminative spectral–spatial features [36]. Classical feature extraction methods include the principal component analysis (PCA) [38], [39], [40], the minimum noise fraction (MNF) [41], and the independent component algorithm (ICA) [42], [43]. Nevertheless, these methods easily yield noisy visual performance since spatial information is not considered. To alleviate this issue, scholars have developed many spectral–spatial feature extraction methods. For example, Benediktsson et al. [44] constructed extended morphological profiles (EMPs) by cascading a series of opening and closing operations. Rasti et al. [45] raised an orthogonal total variation component analysis (OTVCA) to smooth image details, where a low-rank model was used to constrain the spectral dimension. Duan et al. [46] developed a structural profile (SP) to extract discriminative features of HSIs, which has strong robustness to image noise. Zhang et al. [47] proposed the superPCA method to extract global and local spectral–spatial features.

Classifiers aim to assign category labels to the pixels of a given data [48], including spectral classifiers and spectral–spatial classifiers. Spectral classifiers are performed on the raw data to get classification results, such as support vector machine (SVM) [49], [50], decision tree (DT) [51], [52], and random forest (RF) [53]. However, these spectral classifiers cannot obtain satisfactory classification accuracy. In recent years, many spectral–spatial classifiers have been investigated, including mathematical morphology-based spectral–spatial classifiers, sparse representation-based classifiers, and deep-learning-based spectral–spatial classifiers. The mathematical morphology-based spectral–spatial classifiers aim at designing a spectral–spatial classification framework by fusing different levels or types of features. For example, Nicola et al. [54] proposed ICA-based morphological attribute profiles (APs) followed by an SVM classifier. Li et al. [55] developed a subpixel-pixel-superpixel-based multiview active learning classification method, in which a disagreement correlation among different views and a posterior probability prediction were utilized to extend training samples. The sparse representation-based classifiers are to construct a discriminative dictionary and minimize the distance between the test samples and its approximation from each class to determine the class label. For instance, Fang et al. [56] proposed the multiscale sparse representation technique to adaptively extract useful information for the classification of HSIs. Fu et al. [57] designed a shape-adaptive sparse representation model, where a superpixel segmentation method was used to construct shape-adaptive regions. The deep-learning-based spectral–spatial classifiers aim at extracting high-order semantic information followed by a multilayer perceptron to obtain classification results [48]. For example, Hang et al. [58] proposed a two-layer recurrent neural network (RNN) to decrease the redundant and complementary information of HSIs. Zhu et al. [59] designed a spectral attention module (SeAM) and a spatial attention module (SaAM) to improve the characterization ability of HSIs.

The postprocessing methods use the correlation among adjacent pixels to remove the noisy labels. Representative postprocessing classification methods include Markov random fields (MRFs), edge-preserving filtering (EPF), and extended random walker (ERW). For example, Ghamisi et al. [60] proposed a hidden MRF method for spectral–spatial classification of HSIs, in which the MRF model was used to optimize the class probability. Kang et al. [61] presented an EPF-based postprocessing method to improve the classification performance of homogeneous regions. Kang et al. [62] used the ERW method to encode the spatial dependencies of HSIs with a weighted graph.

The presented studies exhibit the huge potential of advanced machine learning methods for land cover mapping. However, it is not clear whether these methods can effectively identify different types of oil films or which one can achieve the best classification performance for oil spill mapping. In this article, we compare the classification performance of several representative classification approaches (e.g., feature extraction, classifier, and postprocessing-based methods) using three case studies. The case studies are comprised of a simulated dataset acquired from an outdoor experiment and two real oil spill accidents that cover different challenges with regard to oil spill characteristics, imaging scenes, and sensors. In all three examples, 12 different classification approaches are selected to evaluate the performance of oil spill mapping in terms of both detection capability and computational efficiency.

The rest of this work is organized as follows. Section II depicts the datasets used in the experiments. Section III presents the used methods in this work. Section IV is devoted to the experiments and results. Section V discusses the influence of different factors. Finally, Section VI presents the conclusions of this work.

---

## II. Datasets

### A. Simulated Dataset

The outdoor experiment was implemented in Qingdao Scientific Research Base, China on September 21, 2020. A large experimental pool of 50 m long, 40 m wide, and 2 m deep was constructed near the coast of the Yellow Sea. The location and overhead view of the experimental pool were shown in Fig. 2. In the outdoor experiment, hyperspectral oil spill images were obtained by a Nano-Hyperspec sensor installed on a DJI-M600PRO UAV. The Nano-Hyperspec sensor uses the push broom mode to capture the simulated dataset. The details of the Nano-Hyperspec and the DJI-M600PRO UAV are listed in Table I.

**TABLE I Details of Nano-Hyperspec Sensor and DJI-M600PRO UAV**
*(Table data omitted in source)*

**Fig. 2. Outdoor experimental site.**

To better explore the potential of HSIs for identifying different types of oil spills and different thicknesses of oil films, we built nine enclosures with polyvinyl chloride (PVC) boards. Each enclosure was 1 m long, 1 m wide, and 1.3 m height. In addition, in order to prevent the oil from leaking to the nearby Yellow Sea, an oil containment boom was placed around the enclosures. In the outdoor experiment, crude oil, gasoline, palm oil, fuel oil, and diesel were selected as research materials of different oil films. In addition, we set the crude oil (1.5, 2.5, and 3.5 mm) and fuel oil (1.0 and 2.0 mm) to explore the performance of different thickness levels of oil films. After even spread of oil films, the Nano-Hyperspec loaded by DJI-M600PRO UAV captured the HSIs in sunny weather. The detailed information of the simulated dataset is shown in Table II. The visualization of the dataset and ground truth is shown in Fig. 3.

**TABLE II Details of the Simulated Dataset**
*(Table data omitted in source)*

**Fig. 3. Simulated dataset. (a) A01 image. (b) Ground truth.**

### B. Penglai Dataset

On June 4, 2011, an oil spill accident occurred in the Penglai 19–3 oil field, which led to massive crude oil and oil-based mud in the ocean. This accident polluted around 6200 km² of ocean area [63], [64]. The site of the Penglai 19–3 oilfield platform is shown in Fig. 4. The Penglai oil spill dataset was obtained by the AISA + hyperspectrometer (Specim Company, Finland) loaded on a sea surveillance aircraft. This dataset has 258 spectral channels ranging from 400 to 1000 nm with a spectral resolution of 5 nm. The study area contains $450 \times 800$ pixels. It should be mentioned that this dataset is contaminated by sun glint. Accordingly, it is difficult to detect oil spill areas for all considered methods. As shown in Fig. 5, the imaging scene contains oil spills, seawater, working ships, and shadow.

**Fig. 4. Site of the Penglai 19–3 oilfield platform.**

**Fig. 5. Penglai dataset. (a) RGB image. (b) Ground truth.**

### C. Deep Water Horizon Dataset

On April 20, 2010, an oil drilling platform named Deepwater Horizon (DWH) operated in the Gulf of Mexico (see Fig. 6) exploded. The accident resulted in the death of 11 workers and was the most serious oil spill accident in history. As a result, about 319 million barrels of oil spilled into the ocean [65], covering an area of more than 20,000 km² [66], [67]. The National Aeronautics and Space Administration Jet Propulsion Laboratory (NASA JPL) used an airborne visible infrared imaging spectrometer (AVIRIS) to collect HSIs. AVIRIS is the first solar reflectance imaging spectrometer in this world invented by JPL with 224 spectral bands from 400 to 2500 nm [68]. Two acquired HSIs are named B01 and B02 with the size of $350 \times 500$ pixels. It can be seen that oil spills in the dataset have different types of states due to weathering and emulsification [69].

**Fig. 6. Location of the DWH platform.**

The Bonn Agreement [70] defines the oil film thickness into five categories, as shown in Table III, according to the visual appearance of oil in a small area. However, due to the low spatial resolution of AVIRIS, a pixel may contain different thicknesses of the oil film. There are three classes of thickness $<50\mu m$ in the Bonn Agreement, but thickness $<25\mu m$ were not detected by Tetracorder (Software developed by USGS to process AVIRIS data) [71]. Moreover, pixels with thickness $>200\mu m$ are only 5% of the total number of pixels but contain more than 45% of the total oil volume [72]. Therefore, according to Sun et al. [72], the Bonn Agreement is modified to three oil film thicknesses, as shown in Table III. The visualization of the dataset and ground truth is shown in Fig. 7.

**TABLE III Bonn Agreement and Thickness Classification in This Study**
*(Table data omitted in source)*

**Fig. 7. DWH dataset. (a) B01 image. (b) Ground truth. (c) B02 image. (d) Ground truth.**

---

## III. Methodology

### A. Feature Extraction Methods

Due to the high spectral dimension, HSI holds a large amount of redundant information. Feature extraction is an effective technique to extract discriminative spectral–spatial features and decreases the number of spectral dimensions. Here, several classical and representative feature extraction methods are adopted to assess the performance of oil spill mapping.

#### 1) PCA:
As a classical and unsupervised feature extraction (UFE) technique, PCA and its variants have been widely applied for HSIs [73], [74]. The PCA algorithm transformed high-dimensional raw data into a low-dimensional space by using a multivariable linear transformation. First, the HSI has converted to a data matrix $I$ after doing zero mean. Then, the eigenvector and corresponding eigenvalue are computed as follows:

$$C=VEV^T \quad (1)$$

where $C$ means the covariance matrix of $I$. $V$ is the eigenvector. $E$ is the eigenvalue. The first three principal components (PCs) are selected to construct image features.

#### 2) Extended Multiattribute Profiles (EMAPs):
EMAP aims to extract the spectral–spatial features with mathematical morphology operation [75]. First, the ICA is used to reduce the number of spectral dimensions. Then, the AP of each component according to (2) is calculated and combined into the extended morphological AP (EAP):

$$EAP=\{AP(FR_1),AP(FR_2),\dots,AP(FR_c)\} \quad (2)$$

where $FR_C$ means $C$th feature after ICA. AP is calculated as follows:

$$AP(L)=\{\phi^T_n(L),\phi^T_{n-1}(L),\dots,\phi^T_1(L),L,\gamma^T_1(L),\dots,\gamma^T_{n-1}(L),\gamma^T_n(L)\} \quad (3)$$

where $\phi^T$ refers to morphological attribute thickening operators. $\gamma^T$ refers to attribute thinning operators. $L$ means the gray-scale image. $T$ means the criterion. EMAP is constructed by the stacked vector approach (SVA), which effectively integrates the global effective feature information.

#### 3) Orthogonal Total Variation Component Analysis:
In [45], OTVCA is a dimension reduction method based on the low-rank model, in which the features of the raw data are calculated by solving the total variation penalized least-squares problem in the following equation:

$$[Equation \ Omitted] \quad (4)$$

where $F$ indicates extracted features. $V$ represents the basis matrix. $Y$ stands for vectorized original image in matrix form. $D_h$ is the operator of first-order vertical, and $D_v$ is the operator of first-order horizontal. $f^{(i)}$ means vectorized feature of the $i$th column of $F$.

#### 4) Multiscale Total Variation (MSTV):
MSTV is an effective feature extraction method, which can well eliminate the image noise and increase the spectral variability of different ground objects [46]. First, an averaging method is used to reduce the number of spectral dimensions. Then, a relative total variation (RTV) is used to construct the multiscale structural features of the dimension-reduced data, which can effectively reduce useless information, such as redundant texture and noise. Finally, KPCA is adopted to fuse the multiscale structural features. The structural features are estimated as follows:

$$\arg\min_S \sum_{i=1}^T (S_i-I_i)^2 + \lambda\cdot\left(\frac{D_x^{(i)}}{L_x^{(i)}+\varepsilon} + \frac{D_y^{(i)}}{L_y^{(i)}+\varepsilon}\right) \quad (5)$$

where $I$ means the input HSI and $S$ means the output structural image. $T$ indexes the number of all pixels. $\lambda$ and $\varepsilon$ are the free parameters controlling the smoothness of structural features and preventing division by zero, respectively. $D_x$ and $L_x$ are the windowed total variations and the inherent variations in the $x$-direction, while $D_y$ and $L_y$ are the windowed total variations and inherent variations in the $y$-direction. More details can be referred to [46] and [77]. In this way, this method not only ensures that the structural features retain the effective information of the original data as much as possible but also confirms that invalid noise and texture are removed.

#### 5) Structural Profiles:
The SP method is intended to retain the main geometrical information and smooth out useless details [77]. The HSI is modeled as a combination of the SP and the texture profile. An adaptive texture smoothing model is designed to extract the significant structural component, and the split Bregman iteration algorithm is used to solve this novel model. The specific model is shown as follows:

$$\arg\min_S \|S-I\|_2^2 \odot \omega + \lambda\|S\|_{TV} \quad (6)$$

where $I$ means the dimension reduced data. $S$ is SP. $\omega$ controls the similarity of pixel neighborhood. $\lambda$ means a free parameter. The specific solution steps and detailed explanation are given in [77].

### B. Classifier

Sparse representation is a classic tool in the field of signal processing, and it has been widely used in all kinds of aspects, such as image restoration [78], image classification [56], and image super-resolution [79]. The raw signal is viewed as a linear combination of several atoms [80], which has been also applied in land cover mapping, image super-resolution [81], target detection [82], and image denoising [83].

#### 1) Multiscale Adaptive Sparse Representation (MASR):
Fang et al. [56] proposed an MASR method for material identification. Specifically, one structural dictionary $D$ and the multiscale matrix $Y_{multiscale}$ are first constructed. Second, in order to employ the correlations among multiscales, MASR designs a new norm $\ell_{adaptive,0}$ to represent a multiscale sparse coefficients’ matrix $A_{multiscale}$ as (7). Third, the sparse coefficients are obtained by iteration:

$$[Equation \ Omitted] \quad (7)$$

where $A_{multiscale}$ is the multiscale sparse matrix and $Y_{multiscale}$ is a multiscale matrix. $D$ indicates a structural dictionary. $K$ means an upper limit of a given sparsity level. Compared with other sparse expression methods, MASR is more accurate in terms of edges and details because of introducing multiscale information.

#### 2) Locality-Constrained Sparse Representation Classifier (LSRC):
Zhang et al. [84] constructed LSRC to improve the classification performance. LSRC assumes that the Euclidean distances between pixels of the same class are also close, so the $k$-nearest neighbor (KNN) algorithm is used to select more dominant atoms from the training samples. This design also reduces the computing complexity of the sparse representation model. To enhance the data separability, linear discriminant analysis (LDA) is used to map the data to a low dimension space. As shown in (8), it is the Euclidean distance $d_n$ between a training sample $x_n$ and the testing sample $y$ in LDA projected space:

$$d_n=\|Ty-Tx_n\|_2^2 \quad (8)$$

where $T$ means the LDA mapping matrix.

Besides, deep learning has become a popular research topic in HSI classification, which can effectively extract high-order semantic information of the input [85]. Various deep learning models have been proposed for the classification of HSIs. In the early stage, researchers proposed some shallow deep networks by using several convolutional layers and pooling layers to extract spatial, spectral, or spatial–spectral features. To improve the classification performance of HSIs, some advanced deep networks and frameworks have been also applied in HSI classification in recent years, such as self-attention mechanisms, self-supervised learning, and active learning. In this work, several representative and advanced deep learning classifiers are selected to test the oil spill detection performance.

#### 3) 2-D Convolutional Neural (2DCNN) Network:
The 2DCNN was applied for land cover mapping [86]. Each layer in the framework is composed of a convolutional layer and a pooling layer. The value of each neuron can be computed as follows:

$$v_{ij}^{xy}=g\left(b_{ij}+\sum_m\sum_{p=0}^{P_i-1}\sum_{q=0}^{Q_i-1}w_{ijm}^{pq}v_{(i-1)m}^{(x+p)(y+q)}\right) \quad (9)$$

where $v_{ij}^{xy}$ means the value at $(x,y)$ of the $j$th channel in the $i$th layer. $w_{ijm}^{pq}$ is the value of weight $(p,q)$ connected to the $m$th channel. $P_i$ is the height of the convolution kernel, and $Q_i$ is the width of the convolution kernel. $b_{ij}$ represents the bias.

#### 4) Autoencoders (AEs):
AE is a representative unsupervised learning network model, including two main structures: encoder and decoder. A simple AE includes one visible layer, one hidden layer, one reconstruction layer, and an activation layer. The encoder maps the input matrix to the hidden layer as (10), while the decoder converts the feature layers to an output layer as (11):

$$Y=f(W_1Z+b_1) \quad (10)$$
$$\hat{Z}=f(W_2Y+b_2) \quad (11)$$

where $Z$ is the input data. $Y$ is the latent feature. $\hat{Z}$ is the reconstruction result. $W_1$ and $W_2$ are the weights. $b_1$ and $b_2$ are the biases.

#### 5) SpectralFormer (SF):
In recent years, the transformer [87] structure applied to natural language processing has attracted widespread attention and successfully developed to image processing [88], [89]. It has been proven that the self-attention mechanism can better extract the global information of data features. The self-attention mechanism can be formulated as follows:

$$z=\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \quad (12)$$

where $Q$, $K$, and $V$ mean vectors generated from input data. The specific explanations and solutions can be referred to [87]. Hong et al. [90] applied the transformer model to HSI classification, called SF. SF utilizes the vision transformer (ViT)-based baseline network for HSIs. The transformer proved that the multihead attention mechanism can better extract the global information of data features. In order to more focus on spectral discrepancies and information connectivity, SF includes a groupwise spectral embedding (GSE) module and a cross-layer adaptive fusion (CAF) module. The former finely captures the local spectral signatures modeled as (13), while the latter fuses the information between layers modeled as (14):

$$\dot{A}=WX \quad (13)$$
$$\hat{z}^{(l)} \leftarrow \ddot{w}[z^{(l)}, z^{(l-2)}] \quad (14)$$

where $A$ means the matrix of feature embeddings. $W$ is the matrix of linear transformations. $X$ is the matrix of spectral signatures. $\hat{z}^{(l)}$ is the fused operation in the $l$th layer. $z$ means the feature. $\ddot{w}$ is the parameter to control fusion.

### C. Postprocessing Methods

#### 1) EPF:
Kang et al. [91] proposed a simple yet effective HSIs’ classification framework with an EPF method. The core idea is to optimize the probability map of each class by considering the correlation in the local regions. Furthermore, the SVM classifier is first performed on the original data to obtain the probability map of each class, and then, the guided filtering is used to remove the scatter pixels:

$$[Equation \ Omitted] \quad (15)$$

In (15) $O$ is the filtered output image. $I$ is the guidance image. $P$ represents the $n$th input image. $i$ and $j$ are, respectively, the $i$th and $j$th pixels. $\omega$ means a local window. $G_{\delta_s}(\|i-j\|)$ and $G_{\delta_r}(|I_i-I_j|)$ are Gaussian decreasing functions. $\delta_s$ defines the size of the local window. $\delta_r$ controls the weight of a pixel decrease. $K_b$ is a normalizing term as follows:

$$K_b^i=\sum_{j\in\omega_i}G_{\delta_s}(\|i-j\|)G_{\delta_r}(|I_i-I_j|) \quad (17)$$

In (16), $a_j$ and $b_j$ are coefficients got by (18), and $\omega$ is the parameter that controls the level of filtering blur:

$$E(a_j,b_j)=\sum_{i\in\omega_j}((a_j I_i+b_j-P_i)^2+\epsilon a_j^2) \quad (18)$$

This classification framework uses the gray image or the color image selected from the original image as the guide image to perform EPF on the classified probability map, which verifies that the local spatial structure information still has a strong improvement effect on the classification accuracy.

#### 2) ERW:
ERW is a graph model-based postprocessing method that aims to optimize the classification map obtained by the spectral classifier. In more detail, the probability map is first obtained by using the SVM classifier. Then, the first PC of HSIs with PCA constructs the weighted graph $G=(V,E)$. $V$ represents pixels in the first PC, and $E$ represents the edges of the first PC. Finally, the ERW is used to optimize the probability map by minimizing the energy function as follows:

$$[Equation \ Omitted] \quad (19)$$

where $p_n$ indexes the probabilities. $L$ means a sparse Laplacian matrix, and the specific solution method is described in [62]. $\Lambda_n$ is a diagonal matrix, and each element on the diagonal is the initial probability of pixels.

---

## IV. Experiments

In order to examine the detection performance of different classification approaches for oil spill mapping, 12 representative classification methods are adopted, including five feature extraction methods (i.e., PCA [92], EMAP [75], OTVCA [45], MSTV [46], and SPs [77]), two sparse representation classification methods (i.e., MASR [56] and LSRC [84]), three deep learning classification methods (i.e., 2DCNN [86], AE [93], and SF [87]), and two postprocessing methods (i.e., EPF [91] and ERW [62]). These methods are highly cited publications and classical techniques in the HSI classification community. Three widely used objective indexes are used to evaluate the classification performance, i.e., average accuracy (AA), overall accuracy (OA), and kappa coefficient (Kappa). For the training set, 1% samples are randomly selected from the reference image.

### A. Parameter Settings
In this section, the parameter settings of all methods follow the original method. Furthermore, we will provide a toolbox. All experiments are done on a computer with 64 GB and an RAM Intel Core i9-10850K CPU and NVIDIA GeForce RTX 3090 GPU. Feature extraction, postprocessing, and sparse representation are implemented computationally on MATLAB R2018a, while the deep learning models are implemented on the PyTorch platform.

*   **PCA:** The number of PCs is the number of classes.
*   **EMAP:** The EMAP method is a representative feature extraction technique [75]. The threshold values of $\alpha$ (area of regions) are set as 200, 500, and 1000. The threshold values of $\delta$ (standard deviation) are set to be 2.5, 5, 7.5, and 10.
*   **OTVCA:** Three parameters are involved in OTVCA [45]. The number of extracted features is set to 16. The parameter that controls the level of smoothness is set as 0.01. The iteration is set as 200.
*   **MSTV:** In [46], the parameters that control the degree of smoothness are set to 0.003, 0.02, and 0.01. The parameters that control the maximum size of texture elements are set to 2, 1, and 3.
*   **SPs:** There are four parameters in SPs [77]. The weight of fusion is 0.5. The number of features after dimension reduction is 30. The smoothing parameter is set to 1.2. The number of kernel PCs is set to 40.
*   **MASR:** In this experiment, the multiscale patch sizes are set to 3, 5, 7, 9, 11, 13, and 15 [56].
*   **LSRC:** There are two parameters in the LSRC algorithm [84]. The sparsity level is 6, and the number of nearest neighbors is set to 20.
*   **2DCNN:** The CNN consists of three blocks. Each block includes a convolution layer, a batch normalization layer, an activation layer, and a pooling layer. The convolutional kernels of three convolutional layers are set to $3 \times 3 \times 32$, $3 \times 3 \times 64$, and $3 \times 3 \times 128$. The patch size, batch size, epoch, and learning rate of 2DCNN are set to 16, 128, 200, and 0.001, respectively.
*   **AE:** For AE [94], the encoder has three fully connected layers with the numbers of neurons of 32, 64, and 128, respectively. Similarly, the decoder has three fully connected layers with the numbers of neurons of 64, 32, and 32. The batch size, epoch, and learning rate of AE are set to 128, 200, and 0.001, respectively.
*   **SF:** The network framework of SF is based on ViT [90]. There are five cascaded encoder blocks. Each block includes a four-head self-attention layer, a multilayer perceptron, and a nonlinear activation layer. The batch size, epoch, and learning rate of SF are set to 64, 500, and 0.0005.
*   **EPF:** There are two parameters in the EPF-based guided filter [91]. The filtering size is fixed to be 3. The value of blur degree is set to 0.01.
*   **ERW:** The optional weighting parameter of the random walker is 710. The seed points are the pixel points of the training set [62].

### B. Simulated Dataset
The first case is studied on the simulated dataset. Fig. 8(a) shows the spectral reflectance of different thicknesses of crude oils. It can be observed that the difference between different thicknesses of oil films is very small. Fig. 8(b) gives the spectral difference of different fuel oils. Accordingly, it is challenging to distinguish them by different classification methods.

**Fig. 8. Spectral curves of different oil films. (a) Crude oil. (b) Fuel oil.**

Fig. 9 shows the oil spill classification results of different methods on the A01 image. The PCA method yields a very noisy oil spill classification map, and different thicknesses of crude oils cannot be distinguished. The EMAP method improves the classification visual effect compared to the PCA method. However, it cannot produce satisfactory classification performance for some oil films, such as gasoline and palm oils. For the OTVCA method, it is unable to accurately identify different thicknesses of crude oils. The MSTV method greatly improves the oil spill classification performance. Nevertheless, the crude oils with 2.5 and 3.5 mm are not well identified. The FDSI method achieves a satisfactory visual classification map. The MASR method yields an obvious block misclassification result. The LSRC method obtains a similar classification result to the PCA method. The 2DCNN method cannot well identify the crude oils with 1.5 and 2.5 mm. The AE method produces unsatisfactory classification performance in identifying different types of oil films. The SF method cannot work well in distinguishing different thicknesses of crude oils. The EPF method yields unsatisfactory classification performance in identifying different oil films. For the ERW method, different types of oil films are well distinguished.

**Fig. 9. Oil spill mapping results of different methods on the A01 image. (a) PCA. (b) EMAP. (c) OTVCA. (d) MSTV. (e) SPs. (f) MASR. (g) LSRC. (h) 2DCNN. (i) AE. (j) SF. (k) EPF. (l) ERW.**

**TABLE IV Classification Accuracies of Different Methods in Percentages for Simulated Dataset. The Best Accuracy in Each Row is Shown in Bold**
*(Table data omitted in source)*

Furthermore, it can be observed that these classification methods can work well in identifying different thicknesses of fuel oils. On the contrary, most of them cannot effectively distinguish different thicknesses of crude oils even under ideal imaging conditions (e.g., solar altitude angle of $54^\circ 09'$ and A05 image). The difference in spectral profiles for three crude oils is very small, especially for 1.5- and 2.5-mm crude oils [see Fig. 8(a)]. Different from the crude oils, the fuel oils with different thicknesses have significant differences in the wavelength ranging from 600 to 900 nm (see Fig. 8), which can be regarded as feature channels.

### C. Penglai Dataset
The Penglai dataset, which is captured from a real accident scene, can better show the performance of different algorithms. Table V shows the objective indexes of different methods on the Penglai dataset, in which the best result is highlighted in bold. For feature extraction methods, the PCA method produces the lowest classification accuracies since this feature extraction technique only considers the spectral information in the original image. The EMAP-based feature extraction method obtains the highest classification performance. For sparse representation methods, both the MASR and LSRC methods fail to work well in classifying different types of objects. For deep learning methods, the 2DCNN method can effectively detect the oil spill region in this dataset corrupted by sun glints among all studied methods. For postprocessing classification methods, the ERW method produces the highest classification accuracies with respect to OA, AA, and Kappa coefficient. Generally, the ERW method still obtains satisfactory classification performance among all considered approaches for this case.

**TABLE V Classification Accuracies of Different Methods in Percentages for the Penglai Dataset. The Best Accuracy in Each Row is Shown in Bold**
*(Table data omitted in source)*

Besides, Fig. 10 shows visual results of different classification methods on the Penglai dataset. The PCA method cannot detect the oil spill region. The EMAP method boosts the classification method. However, there is salt-and-pepper noise in the oil spill region. For the OTVCA and MSTV methods, the oil spills are misclassified into seawater. The SP method produces a noisy classification map for the oil spill and seawater classes. The MASR method cannot identify the oil spill region. The LSRC method also yields a very noisy classification result. The 2DCNN method can well remove the influence of sun glints in the oil spill region and better detect the oil spills. The AE and SF methods produce similar classification results in terms of visual maps. They can detect the oil spill area. However, the classification maps are still very noisy due to the influence of sun glints. The EPF and ERW methods can well remove the salt-and-pepper noise in the classification maps since they can make full use of the spatial correlation among neighboring pixels. On the whole, the ERW method yields the best visual effect in identifying the four types of objects, i.e., ship, oil spill, shadow, and seawater.

**Fig. 10. Oil spill mapping results of different methods on the Penglai dataset. (a) PCA. (b) EMAP. (c) OTVCA. (d) MSTV. (e) SPs. (f) MASR. (g) LSRC. (h) 2DCNN. (i) AE. (j) SF. (k) EPF. (l) ERW.**

### D. DWH Dataset
The third case is studied on the DWH dataset, i.e., B01 and B02 images. In this case, the oil spills have been spread with the movement of seawater, resulting in different thicknesses of oil spills, i.e., “thin” oil, “thick” oil, and “thicker” oil. Tables VI and VII show the objective results of all studied approaches. For feature extraction methods, the PCA method still yields relatively low objective accuracies in this case. The SP method obtains the highest OA, AA, and Kappa coefficient among all feature extraction techniques. For sparse representation methods, the LSRC method is still better than the MASR method in terms of OA, AA, and Kappa coefficient. However, they cannot obtain superior classification accuracies. For deep learning methods, the AE method obtains the best classification performance on the DWH dataset. For postprocessing methods, the ERW method also obtains relatively high classification accuracies in this case.

**TABLE VI Classification Accuracies of Different Methods in Percentages for the B01 Image. The Best Accuracy in Each Row Is Shown in Bold**
*(Table data omitted in source)*

**TABLE VII Classification Accuracies of Different Methods in Percentages for the B02 Image. The Best Accuracy in Each Row Is Shown in Bold**
*(Table data omitted in source)*

In addition, Fig. 11 shows the mapping results on the B01 image. It can be observed that the PCA method ignores the “thick” oil distribution and produces many cluttered and scattered misclassified pixels. The EMAP and OTVCA methods misclassified the “thicker” oil into “thick” oil. The MSTV and EPF methods show the oversmoothing effect on the classification maps. The SPs, LSRC, AE, SF, and ERW methods present better classification maps. Similar to the previous cases, the MASR method still cannot obtain a satisfactory classification effect. It is worth noting that, unlike in the previous cases, the 2DCNN method does not work well in this complex oil spill scene.

**Fig. 11. Oil spill mapping results of different methods on the B01 image. (a) PCA. (b) EMAP. (c) OTVCA. (d) MSTV. (e) SPs. (f) MASR. (g) LSRC. (h) 2DCNN. (i) AE. (j) SF. (k) EPF. (l) ERW.**

---

## V. Model Analysis

### A. Potential of Different Types of Methods
The experimental results clearly show the potential of different types of classification methods. Generally, the spectral–spatial classification methods are always better than the spectral-based classification methods, which illustrates the importance of using spatial correlation of neighboring pixels. For example, the OTVCA method is superior to the PCA method for all cases. In addition, variations of illumination due to imaging conditions and shadows during data acquisition are a challenging problem in oil spill mapping. Most classification methods suffer from performance degradation when the original image is corrupted by sun glint (see case 2).

This study also illustrates the differences in the classification effect of conventional machine learning and advanced deep learning methods. The use of conventional machine learning methods often leads to an oversmoothed phenomenon in the boundaries of different objects due to inappropriate parameters. Moreover, determining how to select the optimal parameter for different scenes is always a challenging problem for all machine learning algorithms. The deep-learning-based classification methods avoid the manual selection of feature extraction methods, which can smartly learn the deep features. By comparing the classification results of machine learning methods and deep learning methods, it is found that the machine learning methods not only outperform the deep learning methods but also exhibit very competitive in terms of classification performance with respect to deep learning methods.

Classification methods remain challenging when faced with oil films of varying thickness, which can be seen from the simulated dataset and the DWH dataset. Compared with other methods, the ERW method performs well in all cases. The reason is that the ERW method makes full use of the spatial correction of local pixels calculated by the graph model. Therefore, the ERW method is more suitable for oil spill mapping.

### B. Influence of Image Noise
In this section, we analyze the influence of image noise on classification performance. Four representative methods, i.e., SPs, AE, LSRC, and ERW, are adopted since they obtain relatively good classification performance. An experiment is performed on the B01 image, in which each band is added by zero-mean Gaussian noise with different variances $\sigma$ set from 0.01 to 0.10 with step 0.01. Fig. 12 presents the objective results of the four considered methods. It is obvious that the classification performance of the ERW method gradually decreases as the noise level increases. Moreover, when $\sigma=0.05$, the classification performance cannot acceptable. The AE and LSRC methods tend to decrease when the original image is corrupted by image noise. By contrast, the SP method is robust to image noise. The reason is that the feature extraction method models the HSI as the combination of the SP (intrinsic property, salient structure, and so on) and the texture profile (e.g., image noise and texture details).

**Fig. 12. Classification accuracy of the B01 image with different variances of zero-mean Gaussian noise. (a) OA. (b) AA. (c) Kappa.**

### C. Influence of Different Numbers of Training Samples
This section is to analyze the influence of different numbers of training samples. Fig. 13 gives the classification accuracies of four considered methods with different numbers of training samples, in which the number of training samples is from 10 to 100 with step 10 for each class. An experiment is performed on the B01 image. The training set is randomly selected from the ground truth, and each experiment is repeated ten times to reduce the influence of the random selected strategy. The reported result is obtained by calculating the average of ten experiments. From this figure, several conclusions can be drawn. First, as the number of training samples increases, the classification accuracies of all methods tend to increase, especially for the ERW method. Second, the LSRC and AE methods still show satisfactory classification performance in the case of limited training samples. For example, when the number of training samples is 10, the LSRC method obtains OA = 83.08%, AA = 81.79%, and Kappa = 0.6434, respectively. Third, as the number of training samples increases, the classification performance of SPs gradually surpasses other methods.

**Fig. 13. Classification accuracy of the B01 image with different numbers of training samples. (a) OA. (b) AA. (c) Kappa.**

### D. Computing Time
Here, we discuss the processing time (in seconds) of all considered classification methods. Fig. 14 depicts the computing time of all methods on three datasets. Several key conclusions can be summarized: First, the computing time of the MASR method is very high because it takes a lot of time to learn multiscale dictionaries. Second, the computing time of deep learning methods takes more time than feature extraction methods. Third, the running time of the postprocessing methods is moderate among all considered approaches since the postprocessing operation is fast. It is worth noting that the computing time of the EMAP method is less than those of other methods for the first two datasets, but it takes a longer time for the third dataset. The main reason is that the EMAP method extracts high-dimension spectral–spatial features from the DWH dataset, which greatly increases the cross-validation time.

**Fig. 14. Computing time of different methods. (a) Simulated dataset. (b) Penglai dataset. (c) DWH dataset.**

---

## VI. Conclusion

HSI classification for oil spill mapping has been a hot topic in the remote sensing field. Most of the current oil spill mapping is limited to detecting a single type of oil species. In this work, we summarize various common methods for hyperspectral oil spill mapping. These methods mainly optimize three steps in the classification process, namely, feature extraction, classifiers including sparse representation and deep learning, and postprocessing. Three datasets, namely, the simulated dataset, the Penglai dataset, and the DWH dataset, are captured or collected to analyze the classification performance. According to the experimental results, several key conclusions can be obtained:

*   The pixel-level feature extraction methods easily produce noisy classification maps. However, the edges of different oil films in the classification maps are more similar to the ground truth. The spectral–spatial feature extraction methods tend to yield oversmoothed classification results because of inaccurate parameter settings.
*   The oil spill mapping performance of sparse representation classifiers overly relies on the constructed dictionary. For example, the MASR method fails to obtain satisfactory performance for oil spill mapping in terms of computational burden and classification accuracies. The deep learning classifiers obtain competitive classification performance among all approaches when the number of training samples is relatively small. Moreover, the deep learning classifiers can automatically learn complex features from HSIs in an end-to-end manner, which has stronger adaptability for different ocean scenes.
*   The postprocessing classification methods can obtain better classification performance when the input HSI is a strong geometric characteristics scene. This is used to the fact that this type of method fully utilizes the correlation among neighboring pixels.
*   Among all considered approaches, the SP-based method has strong robustness to image noise, which is more suitable for oil spill HSIs corrupted by noise. The reason is that this method views the HSI as a combination of an SP (i.e., significant structures and features) and a texture profile (i.e., image noise and useless details), in which the SPs are fed into the spectral classifier.

In the future, it is important to develop some unsupervised, weakly supervised, or self-supervised classification methods for oil spill mapping, so as to overcome the problem of lacking high quality and high amount of training samples. In addition, it is also necessary to develop novel lightweight-oriented models deployed in aircraft systems or orbiting satellites to achieve real-time oil spill mapping.