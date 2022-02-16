# Scene-based Complementary Product Recommendation
#### Abstract
<p align="justify">
In the fashion domain, predicting compatibility is a significantly difficult task due to its subjective nature. Previous work in this domain focuses on comparing product images rather than a real world scene. This fails to capture key context like body type, seasons and other occasions in the scene. This is an important use case which needs to be addressed. "Perfect the Look" which deals with measuring compatibility between a real-world scene and a product. We use two compatibility scores, global and local where global compatibility considers the overall scene and the local compatibility focuses on finer details of the scene using category guided attention mechanism. Different baseline methods were compared with the proposed method and the proposed method gives promising results on Fashion and Home Datasets. This is an implementation of the paper titled "Complete the Look: Scene-based Complementary Product Recommendation"[12].
</p>

#### Methodology
<p align="justify">
	<b>1. Preprocessing</b>: Given the scene image and bounding box surrounding the product, we take the portion of the scene image which appears in the top, bottom, left and right regions of bounding box and consider that region which has the maximum area.
Then, Resize all the images to (256 × 256), apply a RandomCrop (224 × 224), and perform horizontal mirroring for data
augmentation. For training, we have a scene image and a product image that is compatible with the scene. In addition to that randomly select a negative image which is not compatible with the scene. So, a single training instance will be in the form of a triplet i.e (scene, positive product, negative product).</p>
<b>2. Modelling</b>: 
<p align="justify">
	&nbsp;&nbsp;&nbsp;<b>a. Feature Extraction:</b> In this step, the features of the scene and the product images are extracted using the proposed model. The feature map obtained from an intermediate Resnet block is used for the subsequent computation of the embedding.</p><p align="justify">
&nbsp;&nbsp;&nbsp;<b>b. Compatibility Learning</b>: The model focuses on two compatibility scores: local and global.</p><p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;<b>Global</b>: First, the scene and product embeddings are computed from the feature map we obtained in the Feature extraction step. Then, global compatibility is obtained by computing the l2 distance between the two embeddings.</p>
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;<b>Local</b>: 49 regions of size 7 × 7 are used and each patch has a weight associated with it based on it’s relevance with the corresponding product.The local compatibility is computed as the weighted sum ofthe individual compatibility scores between the product embedding and the scene patch. Category guided attention is applied because each category of product might need focus on a specific region with respect to the compatibility. </p>
<p align="justify">	
<b>3. Loss</b>: Finally we compute a hinge loss to train the model so that it learns to decrease the distance between the scene and the positive image and the distance between the scene and the negative image where α is the margin.</p>
<p align="justify">
<b>4. Imagenet Model</b>:
We have used the visual features directly from the ResNet-50 and VGGNet-19 which were pre trained on Imagenet. We can use pre-trained models from the
Pytorch.</p>
	<p align="justify">
&nbsp;&nbsp;&nbsp;<b>a. ResNet-50</b> : Resnet models were proposed in “Deep Residual Learning for Image Recognition”. We have the 5 versions of resnet models, which contains 18, 34, 50, 101, 152 layers respectively.</p><p align="justify">
&nbsp;&nbsp;&nbsp;<b>b. VGGNet-19</b> : VGG19 is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layers, 5 MaxPool layers and 1 SoftMax layer). We can load a pre-trained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.</p><p align="justify">
The similarity is measured using the l2 distance between the scene and product embeddings.</p>
<p align="justify">
<b>5. Siamese Nets Model </b>:
We adopted the Siamese CNNs to learn the embeddings from scene and product images. The following layers (Linear-BN-Relu-Dropout-Linear-L2 Norm) were added at the end of the default Siamese architecture to extract the embeddings in the  desired format. The compatibility between the scene and product was measured using the l2 distance.The model is trained using the triplet loss given in eq , where d is the l2 distance and alpha is 0.2.</p><p align="justify">
<b>6. Attention Weight Model </b>:
This model focuses on thescene to product compatibility which allows the model to make personalised recommendations pertaining to each user. The individual body factors and facial features are considered while suggesting the product to the user. We have focused on another important domain which is the interior design.</p>

![Attention Weights Model](https://user-images.githubusercontent.com/43794593/153901132-39a411fc-581c-4039-8374-ee2d60feab80.png)


#### References
<p align="justify">
[1]  W.-L. Hsiao and K. Grauman, “Vibe:  Dressing for di-verse body shapes,”  inProceedings of the IEEE/CVFConference on Computer Vision and Pattern Recogni-tion, 2020, pp. 11 059–11 069. <br/>
[2]  M. R. Mane, S. Guo, and K. Achan, “Complementary-similarity  learning  using  quadruplet  network,”arXivpreprint arXiv:1908.09928, 2019. <br/>
[3]  D. Cer, Y. Yang, S.-y. Kong, N. Hua, N. Limtiaco, R. S.John,  N.  Constant,  M.  Guajardo-C ́espedes,  S.  Yuan,C.  Taret al.,  “Universal  sentence  encoder,”arXivpreprint arXiv:1803.11175, 2018. <br/>
[4]  S. Hiriyannaiah,  G. Siddesh,  and K. Srinivasa,  “Deepvisual  ensemble  similarity  (dvesm)  approach  for  visu-ally aware recommendation and search in smart com-munity,”Journal of King Saud University-Computerand Information Sciences, 2020. <br/>
[5]  Y.-L.  Lin,  S.  Tran,  and  L.  S.  Davis,  “Fashion  outfitcomplementary item retrieval,” 2020. <br/>
[6]  W.-C.  Kang,  C.  Fang,  Z.  Wang,  and  J.  McAuley,“Visually-aware  fashion  recommendation  and  designwith  generative  image  models,”  in2017 IEEE Inter-national Conference on Data Mining (ICDM).    IEEE,2017, pp. 207–216. <br/>
[7]  H. Zhang,  X. Yang,  J. Tan,  C.-H. Wu,  J. Wang,  andC. C. J. Kuo, “Learning color compatibility in fashionoutfits,” 2020. <br/>
[8]  X.  Li,  X.  Wang,  X.  He,  L.  Chen,  J.  Xiao,  and  T.-S. Chua, “Hierarchical fashion graph network for per-sonalized  outfit  recommendation,”  inProceedings ofthe 43rd International ACM SIGIR Conference on Re-search and Development in Information Retrieval, 2020,pp. 159–168. <br/>
[9]  “https://github.com/kang205/stl-dataset.” <br/>
[10]  Y.  Lin,  M.  Moosaei,  and  H.  Yang,  “Outfitnet:  Fash-ion outfit recommendation with attention-based multi-ple instance learning,” inProceedings of The Web Con-ference 2020, 2020, pp. 77–87. <br/>
[11]  W. Li and B. Xu, “Aspect-based fashion recommenda-tion with attention mechanism,”IEEE Access,  vol. 8,pp. 141 814–141 823, 2020. <br/>
[12]  Kang, Wang-Cheng and Kim, Eric and Leskovec, Jure and Rosenberg, Charles and McAuley, Julian,  “Complete the Look: Scene-Based Complementary Product Recommendation,” inProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. <br/>
</p>

#### Team Members
Shreya Goel <br/>
Roshan S <br/>
Anubhav Ruhela <br/>
Richa Dwivedi <br/>
Dharmendar Kumar <br/>
