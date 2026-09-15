# Self-Supervised Spectral–Spatial Transformer Network for Hyperspectral Oil Spill Mapping
Xudong Kang; Bin Deng; Puhong Duan; Xiaohui Wei; Shutao Li

## Introduction

With the development of economic globalization, marine transportation industries are developing rapidly, and marine oil spill accidents have occurred frequently, which dumps many harmful substances into the ocean [1], [2]. In 2010, the explosion of the Deepwater Horizon platform led to 560,000 tons of crude oil leaking into the Gulf of Mexico. This accident caused a devastating effect on the biological resources and marine environment. In 2021, two cargo ships collided near the Chinese port city of Qingdao, spilling about 9,400 tons of cargo oil into the sea. The oil spill affected a total area of 4,360 km² and a coastline of 786.5 km. It will take more than ten years for the natural fishery resources in the accident area to recover to the prepollution level. Oil spill pollution has become one of the most serious and complex problems of marine pollution in the world. After the oil spill accident, different kinds of oil spills need to be treated in different ways. Therefore, detection and identification of different oil spills is an important technical support for emergency command work and oil pollution clean-up.

Remote sensing has become a prevailing technique for oil spill monitoring, such as multispectral images, synthetic aperture radar, and hyperspectral images (HSIs) [3], [4], [5], [6], [7], [8], [9]. Compared with other remote sensing techniques, HSIs not only provide spatial information but also have a higher spectral resolution, which has attracted extensive attention in recent years [10], [11], [12], [13], [14], [15], [16], [17]. Many hyperspectral oil spill detection methods have been studied, which can be roughly divided into three categories: hand-crafted feature-based methods, spectral-classifier-based methods, and deep-model-based methods.

Hand-crafted feature extraction-based approaches aim to extract shallow features from the input for the spectral classifier. For example, Dongmei et al. [18] proposed a wavelet-transform-based oil film classification method. The singularity of the high-frequency coefficient was used to select the sensitive bands for different thicknesses of oil spill film. Lu et al. [19] and [20] used a decision tree method for distinguishing water in oil and oil in water emulsions from seawater. Duan et al. [21] first constructed a hyperspectral oil spill detection dataset and proposed an unsupervised detection method with promising detection performance. However, these methods seriously depend on the selected bands, which are often very complicated and time-consuming. Therefore, how to fully use a wealth of spectral information in HSIs is the key to accurately identifying and classifying the oil spill.

The spectral-classifier-based methods aim to measure the spectral similarity between a given spectrum and all the candidate spectra by traditional machine learning methods. For example, Löw et al. [22] used the random forest method for oil spill mapping. Liu et al. [23] proposed a minimum noise fraction transform-based decision tree classification method to extract the relative thickness of the oil film. In ocean scenes, sun glints in HSIs are unavoidable. Therefore, the sun glints are first removed, and then, the spectral classification method is used to identify different types of oil films. For example, Yang et al. [24] used a multiscale wavelet-transformed to remove the sun glints in HSIs, and then, the obtained results were fed into a deep neural network classifier. Duan et al. [25] proposed a texture-aware total variation method to eliminate the sun glints, and then, the support vector machine (SVM) method was used for oil spill classification [24], [25]. These methods only consider the spectral information, easily yielding noisy classification results.

In recent years, deep learning methods have also been applied for the oil spill classification of HSIs [26], [27], [28], [29]. For example, Zhu et al. [30] proposed a convolutional neural-network-based oil film thickness classification method and demonstrated the effectiveness of the deep learning method for oil spill mapping. Liu et al. [31] proposed a spectral index band selection method and used a 1-D convolutional neural network to realize automatic detection and identification of oil spills. Wang et al. [32] proposed a spectral–spatial feature fusion network for hyperspectral oil spill classification. First, the spectral–spatial features are extracted with 1-D and 2-D convolutional neural networks. Then, the extracted spectral–spatial features are fused. Finally, the fused features are sent to the classifier for classification [32]. These deep-learning-based methods can improve the classification performance with sufficient training samples. However, it is difficult to label a larger number of training samples in marine oil spill scenarios.

Recently, transformer networks have been widely used in the signal processing field since they are based on the self-attention mechanism to make the network focus on the more informative region of the input data [33], [34], [35], [36]. Furthermore, the transformer architecture has a powerful modeling ability of long-range dependencies and strong generalization capability, which provides huge potential for hyperspectral oil spill mapping [37]. In this study, we propose a self-supervised spectral–spatial transformer network (SSTNet) for hyperspectral oil spill classification, which mainly consists of two key steps, i.e., the pretext and the downstream tasks. First, in the pretext task, the original data and augmented data are fed into a transformer-based contrastive learning network to extract deep features. Then, the learned parameters are transferred to the downstream classification network to learn the classifier. Finally, the learned classifier is performed on the input data to obtain the oil spill classification result. Experiments on the hyperspectral oil spill database (HOSD) show that our method achieves superior classification performance with respect to other advanced approaches. 

The main contributions of this work are as follows:
* A self-supervised SSTNet is proposed for hyperspectral oil spill mapping. To the best of our knowledge, this is the first work to investigate the self-supervised transformer network for hyperspectral oil spill mapping.
* A spectral–spatial pooling module (SSPM) is proposed to expand the receptive field and reduce the computational cost, and a spectral–spatial residual module (SSRM) is designed to enhance the discrimination of different types of objects by modeling high-order semantic information.
* We establish a hyperspectral oil spill mapping database, named HOSD. Experiments on the built HOSD prove that our method is superior to other representative methods.

This article is organized as follows. Section II describes the specific steps of the proposed method in detail. Section IV discusses the experimental results. The conclusions are given in Section V.

---

## II. Proposed Method

*Fig. 1. Flowchart of the proposed hyperspectral oil spill mapping method.*

Fig. 1 depicts the flowchart of the proposed oil spill mapping method, which consists of two key steps. First, the deep features of original images are extracted with a self-supervised spectral–spatial transformer in the pretext task. Then, the pretrained spectral–spatial transformer is transferred into the downstream classification task to obtain the classification results.

### A. Overview of Self-Supervised Spectral–Spatial Transformer

Current deep networks can obtain satisfactory classification performance when the number of training samples is sufficient. However, it is hard to annotate a mass of labels in real applications. In this work, to overcome this problem, a self-supervised spectral–spatial transformer is proposed. First, a data augmentation scheme is used to improve the diversity of training data, which mainly consists of image noise, spatial mirror, and spectral rotation. 

Assume $I$ to be the input HSI, random noise is added into the input data $I$ to obtain the noisy data $\tilde{I}$, and a rotation operation is performed on the noisy data $\tilde{I}$ so as to produce spatial rotated data $\{\tilde{I}_\theta | \theta \in \Theta\}$, where $\Theta=\{90^\circ \cdot t | t \in [0,1,2,3]\}$ is the rotation degree. In the spectral domain, a spectral rotation technique [16], [38] is used to enhance the diversity of training data by rotating the hyperspectral data cube along with the spectral axes. The reason is that the spectral rotation technique increases the amount of training samples and makes the deep network better learn the commonality of the same class. Specifically, a spectral rotation is performed on the rotated data $\tilde{I}_\theta$ to obtain the spectral rotated data $\tilde{I}_{\theta_1}$, and thus, its spectral sequence in $\tilde{I}_{\theta_1}$ is arranged from high frequency to low frequency. Accordingly, the augmented data are expressed as $X=\{\tilde{I}_\theta \bigcup \tilde{I}_{\theta_1}\}$. In the training stage, the augmented data $X$ are randomly chosen as the input of the pretext task.

Then, the pretext task aims at learning the deep embedding features from the original image and its augmented image without any labels. Given a hyperspectral cube $x_i^a$ that is cropped from the original image $I$ with a fixed window $W$ and its corresponding augmented cube $x_i^b$ cropped from the augmented data $X$. Assume $\Gamma(\cdot, \varphi)$ to be the encoder network that is the proposed spectral–spatial transformer parameterized by $\varphi$. The deep features $f_i^a$ and $f_i^b$ of unlabeled samples $x_i^a$ and $x_i^b$ can be obtained by the spectral–spatial transformer $\Gamma(\cdot, \varphi)$. To obtain the deep features from unlabeled samples, the training strategy is that the gap between similar features is smaller, and the different features are farther away from each other. A contrastive loss is adopted as follows:

$$
L_i = -\log \frac{\exp(f_i^a \cdot f_j^b / \tau)}{\sum_{k=1}^N \exp(f_i^a \cdot f_k^b / \tau)} \quad (1)
$$

where $\tau$ is a temperature parameter. In this work, $\tau$ is set as 0.07 via empirical validation [39]. 

To learn the deep features from the original images, a scalable hyperspectral cube should be fed into the spectral–spatial transformer. However, the network easily suffers from a larger mini-batch optimization problem during the training stage for the larger batch size. To solve this issue, a momentum bank mechanism is used to accumulate the embedding features, which act as negative samples. Specifically, a queue of deep features of image cube $x_i^a$ is built, in which the size of the queue is larger than the one of the minibatch to increase the learning ability of the network. In the training stage, the deep features of current mini-batch are compared with the ones in the queue, where the features of current minibatch are enqueued and the oldest ones are dequeued. To update the deep features in the queue, an auxiliary spectral–spatial transformer with parameter $\varphi_{aux}$ is introduced as the momentum encoder, in which the update rule is as:

$$
\varphi_{aux}^{(t+1)} = m\varphi_{aux}^{(t)} + (1-m)\varphi^{(t)} \quad (2)
$$

where $m \in [0,1)$ is the momentum coefficient. In this work, $m$ is set to be 0.999.

Finally, the parameters learned from the pretext task are transferred to the downstream classification network. A small amount of training set is fed into the classification network, which is solved by minimizing the classification objective function:

$$
Loss = -\frac{1}{C} \sum_{i=0}^C (y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)) \quad (3)
$$

where $C$ is the total number of classes, $y_i$ is the ground truth, and $\hat{y}_i$ is the predicted label.

### B. Spectral–Spatial Transformer Network

To extract the discriminative features from complex ocean scenes, an SSTNet is designed, which mainly consists of two key modules, i.e., SSPM and SSRM.

**1) Spectral–Spatial Pooling Module:**
The SSPM consists of a pooling transformer and a spectral transformer with a projection shortcut connection. *Fig. 2(a)* shows the structure of SSPM. The SSPM uses a pooling transformer to reduce the spatial resolution of the input feature maps so as to decrease the computational costs. Moreover, this way allows the output features to get a bigger receptive field. The calculation formula of the SSPM is as follows:

$$
Y = T_S(T_P(X)) + Pool(X) \quad (4)
$$

where $X$ is the input feature, and $Y$ is the output feature. Here, $Pool$ represents a max pooling operation, and its pooling kernel is $3 \times 3$. $T_P(\cdot)$ indicates the pooling transformer, and $T_S(\cdot)$ denotes the spectral transformer. The detailed structures of the pooling transformer and the spectral transformer are as follows.

*Fig. 2. Structure of two types of spectral–spatial module. (a) SSPM. (b) SSRM.*

**a) Pooling transformer:**
The high computational cost is one of the most challenging problems in vision transformer. A common solution to this problem is to use the pooling operation in the transformer network to reduce the feature dimension. However, the pooled features extracted by a single pooling operation seem to be less powerful [40]. To solve this problem, a pooling transformer is developed. *Fig. 3(a)* shows the structure of the pooling transformer. Specifically, let $X \in \mathbb{R}^{H \times W \times C}$ ($H$, $W$, and $C$ are the height, width, and channel dimension of HSI, respectively) be the input feature. The queries $Q$, keys $K$, and values $V$ are defined as:

$$Q, K, V = X \quad (5-7)$$

where $V$ is achieved through the $1 \times 1$ convolution, and $K$ is achieved through the $3 \times 3$ convolution. $W_V$ and $W_K$ are trainable parameters. Then, we downsample $Q$, $K$, and $V$ in the spatial transformer to get $Q_P$, $K_P$, and $V_P$:

$$
Q_P = Pool_Q(Q), \quad K_P = Pool_K(K), \quad V_P = Pool_V(V) \quad (8)
$$

where $Pool$ represents the max pooling operation. Next, the attention matrix $A_P \in \mathbb{R}^{h \times w \times (3 \times 3 \times C)}$ ($h$ and $w$ are the height and width of the pooling result, respectively) can be obtained through:

$$
A_P = [K_P, Q_P]W_{P1}W_{P2} \quad (9)
$$

where $W_{P1}$ and $W_{P2}$ are the $1 \times 1$ convolution operation, and their stride and padding are 1 and 0, respectively. The output channel $D$ of $W_{P1}$ is the largest integer not greater than $(C/2)$. Finally, the output feature of the pooling transformer is estimated as $Out_P \in \mathbb{R}^{h \times w \times C}$:

$$
Out_P = V_P \times A_P + K_P \quad (10)
$$

*Fig. 3. Architecture of three transformers. (a) Pooling transformer. (b) Spectral transformer. (c) Spatial transformer.*

**b) Spectral transformer:**
As shown in *Fig. 3(b)*, the spectral transformer aims to extract the contextual feature across spectral bands from the input. Specifically, the queries, keys, and values of the spectral transformer are first calculated as:

$$Q_S, K_S, V_S = X \quad (11-13)$$

where $Conv_{3D}$ represents a 3-D convolution operation, and its kernel_size, stride, and padding are [7, 1, 1], [1, 1, 1], and [3, 0, 0], respectively. Then, the attention matrix of the spectral transformer $A_S \in \mathbb{R}^{C \times C}$ is obtained:

$$
A_S = Q_S \times K_S \quad (14)
$$

Finally, the output feature $Out_S \in \mathbb{R}^{H \times W \times C}$ is estimated as:

$$
Out_S = V_S \times A_S + K_S \quad (15)
$$

**2) Spectral–Spatial Residual Module:**
The SSRM consists of a spatial transformer and a spectral transformer with a shortcut connection (see *Fig. 2(b)*). The input features pass through a spatial transformer and a spectral transformer and then add the original input features to obtain the output feature:

$$
Y = T_S(T(X)) + X \quad (16)
$$

where $T(\cdot)$ denotes the spatial transformer, in which the spectral transformer has been described mentioned above. Here, the spatial transformer is introduced as follows.

**a) Spatial transformer:**
Most transformers do not take advantage of the rich context between adjacent keys [41]. Therefore, we adopt the contextual spatial transformer to extract spatial features, which fully capitalizes on the contextual information among input keys to guide the learning of the dynamic attention matrix. The structure of the spatial transformer is shown in *Fig. 3(c)*. Specifically, let $X \in \mathbb{R}^{H \times W \times C}$ be the input feature. The queries $Q$, keys $K$, and values $V$ are obtained by a convolution operation. Then, the attention matrix can be obtained via two consecutive $1 \times 1$ convolutions ($W_1$ and $W_2$):

$$
A = [K, Q]W_1W_2 \quad (17)
$$

where $A \in \mathbb{R}^{H \times W \times (3 \times 3 \times C)}$ is the attention matrix. Finally, the output feature $Out \in \mathbb{R}^{H \times W \times C}$ is calculated as:

$$
Out = V \times A + K \quad (18)
$$

---

## III. Experimental Setup

### A. Database
To test the effectiveness of the proposed method, a hyperspectral oil spill mapping database is constructed, named as HOSD. This database is collected from the AVIRIS sensor over the Gulf of Mexico crude oil spill area, which contains eight HSIs (HOSD1-HOSD8). Each HSI has 144 spectral bands ranging from 0.4 to $2.5\mu m$. The spatial size of each image is $500 \times 350$ pixels. The spatial resolution of each image is different due to different flight heights, which is shown in **TABLE I Information for the HOSD**. 

*Fig. 4. Visualization of false-color composite images and reference images of the HOSD. The reference images are obtained by domain experts and other publications [32], [42], [44]. (a) HOSD1. (b) HOSD2. (c) HOSD3. (d) HOSD4. (e) HOSD5. (f) HOSD6. (g) HOSD7. (h) HOSD8.*

According to the Bonn Agreement Oil Appearance Code [42], [43], this database is labeled into four classes: thick oil, thin oil, sheen, and seawater.

### B. Parameter Settings
In the work, the stochastic gradient descent (SGD) optimizer with a minibatch size of 256 is adopted. The spatial size of each cropped cube is 11. In the pretext task, the learning rate is initialized with 0.01 and updated by multiplying 0.1 when the epoch times are 120 and 200. The total number of epochs is set to be 280. In the classification task, the learning rate is initialized with 0.03 and decayed by multiplying a scale of 0.1 when the epoch time is 30. The number of epochs is set to be 50. To better train the classifier, 5% training samples selected from HOSD1 and HOSD2 images are adopted. All the experiments were run on a personal computer using the PyTorch platform. The specific details are shown in **TABLE II Hardware and Software Configuration of Experimental**.

### C. Evaluation Indexes
Four widely used evaluation indexes, i.e., overall accuracy (OA), average accuracy (AA), Kappa coefficient, and $F_1$ score, are used to quantitatively evaluate the performance of different classification methods [45], [46], [47], [48], [49]. OA is the ratio of all the correctly classified samples in the total labeled samples. AA represents the mean of the percentage of correctly classified for each class. The Kappa coefficient is an indicator for consistency testing to measure the performance of the classification. The $F_1$ score is defined as the harmonic mean of precision and recall. The larger the four classification indicators, the better the classification effect.

### D. Comparison Methods
To demonstrate the advantage of our method, four types of classification comparison methods are adopted:

*   **Traditional Machine Learning Methods:** SVM [50] and multiscale total variation (MSTV) [51].
*   **Convolutional Neural-Network-Based Methods:** Deep residual network (Resnet50) [52], 3-D convolutional neural network (3DCNN) [53], and spectral–spatial residual network (SSRN) [54].
*   **Transformer-Based Method:** SSTN [36].
*   **Self-Supervised Learning (SSL) Oil Detection Method:** SSL [16].

The number of training samples is 5%, which is selected from HOSD1 and HOSD2.

*   **SVM:** The SVM method is a classical spectral classifier, which is implemented on the SVM library using a Gaussian kernel with fivefold cross validation.
*   **MSTV:** The MSTV method is a relative total variation-based multiscale structural feature extraction method. The smoothing parameters are set to be 0.003, 0.02, and 0.01. The spatial scale is set to be 1, 2, and 3.
*   **Resnet50:** The Resnet50 method is a representative residual network.
*   **3DCNN:** The 3DCNN method uses 3-D convolutions to extract both the spectral and spatial features.
*   **SSRN:** The SSRN is a method with residual blocks, which uses 2-D convolutions to extract spatial features and 3-D convolutions to extract spectral features.
*   **SSTN:** The SSTN builds on SSRN by replacing the 2-D convolution with the spatial transformer and the 3-D convolution with the spectral transformer.
*   **SSL:** The SSL is a self-supervised training method with similarity loss for hyperspectral oil spill detection.

---

## IV. Experimental Results

### A. Ablation Study
To verify the effectiveness of the proposed modules, i.e., SSPM, SSRM, and SSL (pretext), an experiment is performed on the HOSD. *Fig. 5* gives the classification accuracies of the proposed method with or without these modules. It can be observed that the proposed method without the SSPM yields lower classification accuracies than the proposed method. The main reason is that the proposed method fails to fully use the global information when the SSPM is not adopted in the proposed framework. The SSRM mainly aims to extract the spectral–spatial features of HSIs, which plays an important role in the feature extraction process. It can be seen from *Fig. 5* that the proposed method with the SSRM obtains the lowest classification accuracies. Furthermore, when the SSL technique is not used in the proposed framework, the classification performance of the proposed method tends to dramatically drop. This is due to the fact that SSL can effectively enhance the robustness of this method.

*Fig. 5. Classification accuracies of the proposed method with or without the proposed modules. (a) OA. (b) Kappa.*

### B. Objective Comparison
**TABLE III Classification Results (%) of Different Methods on the HOSD** shows the objective results of different methods on the HOSD. The SVM method can obtain satisfactory objective accuracies on HOSD1 and HOSD2. However, it suffers from a marked drop in other samples since the spectral classifier has low transfer capability. The MSTV method slightly improves the classification accuracies. However, it still cannot perform well for some complex scenes, such as HOSD4, HOSD5, HOSD7, and HOSD8. The ResNet 50 yields a similar classification performance to the MSTV method. The 3DCNN method obtains worse classification accuracy for the HOSD5 image. The SSRN method produces moderate classification accuracy among all the considered methods. The SSTN method cannot obtain high classification results on some images, such as HOSD5 and HOSD7. The reason is that this method is a supervised classification method learned from HOSD1 and HOSD2 images, which has poor adaptation for other images. The SSL method yields relatively stable classification performance on all the images since it is an SSL method. As expected, the proposed method achieves the highest objective results on all the images, and the average objective metrics, i.e., OA, AA, Kappa coefficient, and $F_1$ score, are much higher than other methods. This is quite reasonable since a self-supervised spectral–spatial transformer is designed to extract deep discriminative features from massive unlabeled images.

### C. Visual Comparison
*Fig. 6* shows the visual classification maps of all the methods on the HOSD. As shown in this figure, the SVM method obtains an unsatisfactory visual effect, containing “salt and pepper” noise in the classification maps. Moreover, the oil films cannot be well-identified for most of images. The MSTV method obtains satisfactory classification performance on HOSD1 and HOSD2 images. However, it yields serious misclassification in the visual maps for other images since this method is only trained on HOSD1 and HOSD2 images. The ResNet50 and 3DCNN methods hardly identify the oil films for HOSD5 image. The reason is that the appearance of HOSD5 image is different from that of other images, especially for the “Seawater and Sheen” classes. For the SSRN method, the “Seawater” class is classified into the “Sheen” class due to the spectral similarity of the two classes. The SSTN method produces an over-smoothed phenomenon. For the SSL method, the boundaries of different objects are not well-aligned with the reference image. In contrast, the proposed method obtains the best visual maps for all the images in identifying different types of oil films.

*Fig. 6. Classification maps obtained by different methods on the HOSD. (a) SVM. (b) MSTV. (c) ResNet50. (d) 3DCNN. (e) SSRN. (f) SSTN. (g) SSL. (h) Our method.*

### D. Computing Time
Since the spatial size of all the sample images is consistent, we only take HOSD1 as an example to compare the computing time. Considering that the SVM and MSTV methods are performed on the MATLAB software, we only compare the computing time of different deep learning methods. **TABLE IV Computing Time of Different Methods on the HOSD** shows the training and test times of different deep learning methods. It can be observed that the SSL methods perform the slowest with respect to other deep networks since the computational efficiency in the pretext task is high. In the downstream task, the ResNet50 and SSL methods cost more execution time since the two methods use ResNet50 deep network as backbone. In the test phase, the proposed method has a lower inferring time. This is due to the fact that the proposed method designs a pooling transformer to reduce the feature dimension. In general, when the network parameters have been learned offline, the implementation efficiency of the proposed method is superior to other deep networks.

---

## V. Conclusion

In this article, a self-supervised SSTNet is proposed for oil spill mapping of HSIs. The proposed method is mainly composed of two important steps, i.e., the pretext task and the downstream task. The pretext task is to learn the significant features from original images without any labels. Then, the learned parameters in the pretext task are transferred to the downstream classification network, which only needs to be trained with a small amount of labeled data in the downstream task. Experiments on the HOSD demonstrate that our method is superior to other studied methods in both the objective and subjective results, even when the number of training samples is limited. Moreover, the computational burden can be greatly reduced. We have to admit, however, that despite a significant performance improvement in the proposed method, its transfer ability is still limited for real-world applications, especially when the test dataset is from other sensors. In the future, we will study unsupervised domain adaptation methods for hyperspectral oil spill mapping to further strengthen the transfer ability.

### ACKNOWLEDGMENT
The authors would like to thank the editors and anonymous reviewers for their insightful comments and suggestions, which have significantly improved this article.