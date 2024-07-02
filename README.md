# NDA-Detector

## Introduction
This repository is a pytorch implementation of the paper "NDA-Detector: Non-local dual attention detector for thermal protection material defect detection", the code is a copy from the private repository of our engineering project, the code will be made public after removing sensitive information.

## Abstract
Thermal protection materials are widely used in the aerospace field. Existing generic object detection models have defect detection difficulty in thermal protection materials due to the similarity between defects and background, tiny area and multi-scale characteristics. This paper proposes a novel non-local dual attention(NDA)-detector, which achieves accurate and real-time defect detection for thermal protection materials. First, skip connection, atrous convolution and adaptive average pooling are employed to improve the texture enhanced module to enhance concealed defect textures and features. Second, the proposed non-local dual attention(NDA) addresses the problem of severe loss of features for tiny defects. Finally, the path aggregation network fuses the NDA improves the detector's ability to detect multi-scale defects. The experiments on the presented digital radiography dataset show that our detector obtains a 54.74% mAP@0.5 with at least 25 frames per second. Compared to the original detector used in the evaluation, mAP@0.5 is improved by 11.05%. Furthermore, a publicly available dataset was utilized to verify the effectiveness of the proposed method. Thus, NDA-detector exhibits considerable potential in the field of defect detection for thermal protection materials.

## Timeline:
Project creation date: 5/8/2023

Creation of this repository: 11/14/2023

Article Submitted: 5/7/2024

Project Status Display: 7/2/2024

## Project Status
We have built the thermal protection material defect AI recognition platform based on the NDA-detector model, and we now publish the details of the platform as follows.

Figure 1 shows the technology stack of the platform
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E5%B9%B3%E5%8F%B0.png)

Figure 2 shows the flowchart of the platform
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E5%B9%B3%E5%8F%B0%E6%B5%81%E7%A8%8B2.png)

Figure 3 demonstrates the data flow diagram of the platform
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E6%97%B6%E5%BA%8F%E5%9B%BE.png)

Figure 4 demonstrates the data annotation process of the platform (based on CVAT)
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E6%96%B0%E7%89%88%E6%9C%AC.png)

Figure 5 shows the basic interface of the platform
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E6%9C%BA%E5%99%A8%E6%A3%80%E6%B5%8B.jpg)

Figure 6 dynamically shows the platform's machine detection process
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E6%9C%BA%E5%99%A8%E6%A3%80%E6%B5%8B.gif)

Figure 7 dynamically shows the platform's manual re-inspection process
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E4%BA%BA%E5%B7%A5%E5%A4%8D%E6%A3%80.gif)

Figure 8 dynamically shows the platform re-labeling process
![](https://github.com/ChialinSung/NDA-Detector/blob/main/show_images/%E4%BA%8C%E6%AC%A1%E6%A0%87%E8%AE%B0.gif)

More details will be revealed in the future, so stay tuned!
