# Detection-Tutorial

#### 👉 這是一個以「 Detection 」為中心所釋放出的函式庫，其中包括現今各種 Detection Model，像是：__SSD__ ...

#### 👉 此 Repository 有系統地將模型架構拆解成不同元素，同時提供模型相關的 __Config File__、__Functional Python Files__ ... ，以便讓使用者使用相關套件或是函式

## 目錄

- [特色介紹](#features_introduction)

    + [框架介紹](#framework)

    + [框架特色](#features)

- [模型種類及介紹](#model)

    + [支援模型](#support_model)

    + [各模型介紹](#model_introduction)

- [環境設置及套件安裝](INSTALL.md#environment)

- [快速教學指南](GETTING_STARTED.md#guide)

    + [了解目錄樹](GETTING_STARTED.md#directory_tree)

    + [如何訓練以及測試](GETTING_STARTED.md#train_test)

    + [基本設定檔](GETTING_STARTED.md#config)

- [參考資料](#reference)

<a name="features_introduction"/>

## 特色介紹

<a name="framework"/>

- ### 框架介紹
    + 此 Repository 參考 [__MMDetection__](https://github.com/open-mmlab/mmdetection) 以及其 [__論文__](https://arxiv.org/abs/1906.07155) 來實作而成。
    + 其將 Detection 的 [__模型__](Model/) 分為：
        - BackBone
        - Neck
        - Anchor Head
        - ROI Extractor
        - ROI Head
        - ...
    + 此外，也包括其他 [__Utils__](Utils/) ：
        - Loss
        - NMS
        - Calculate_mAP
        - Decode / Encode Coordinates
        - ...
    + 在此架構之下，任何模型都可以拆解成不同元素，並且也可以交互用在不同 Detection 模型中。

        __=> 可以將每個構成模型的元素都看成一個個物件，並且可以依照自己所想要的架構將所要的物件組合起來，利用  Config File 來完成訓練與測試__

<a name="features"/>

- ### 框架特色

    + 由上面框架介紹可知，此架構的設計讓所謂的 __模型模組化__，此外也同時能 __滿足不同框架的 Detection Model__。

    + 引用自[__Major Features__](https://github.com/open-mmlab/mmdetection#major-features)：

        ```
        * Modular Design

            We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

        * Support of multiple frameworks out of box

            The toolbox directly supports popular and contemporary detection frameworks, e.g. Faster RCNN, Mask RCNN, RetinaNet, etc.

        * High efficiency

            All basic bbox and mask operations run on GPUs now. The training speed is faster than or comparable to other codebases, including Detectron, maskrcnn-benchmark and SimpleDet.

        * State of the art

            The toolbox stems from the codebase developed by the MMDet team, who won COCO Detection Challenge in 2018, and we keep pushing it forward.
        ```
        __✨ 可看出此架夠完全的讓開發人員很迅速的專注在模型設計上，而不是架構的搭建__


<a name="model"/>

## 模型種類及介紹

<a name="support_model"/>

- ### 支援模型

    + 目前只支援 __Single-Stage Detector (SSD)__，正在添加其他 Model、BackBone、Neck ...等其他模組，主要包括 __Faster - RCNN__，預計在年底前完成

        |   | ResNet | VGG |
        |:---:|:---:|:---:|
        | SSD | ✘ | ✓ |
        | Faster - RCNN | ✘ | △ |

<a name="model_introduction"/>

- ### 各模型介紹

    - __Single-Stage Detector (SSD)__ :

        + __示意圖__ ：

            ![SSD](https://i.imgur.com/8NTyiZw.png)

        + 在此架構之上，預設的模型包括了：

            - BackBone : VGG-16 Net。此外 SSD 將 Fully-Connected Layer 的兩層換成完全卷積層，並且延伸到 Conv_11 曾，總約 20 多層

            - Anchor Head : 在此部分，SSD 會在這根據所定義的 __numbers of anchor__ 來定義 __Predict Layers__，並且一樣用卷積層來輸出預測的值，包括：__座標、類別分數__

            - Prior Boxes : 這邊我們利用所定義的 __Feature Maps__ ，同時根據定義的 __Min_size / Max_size__ 來依序生成所對應的 Prior Boxes (Anchor Boxes) -- __8732個__ Prior Boxes

            - Loss Functions : 
                
                ![loss](https://i.imgur.com/UeQj0XP.png)

                * Confidence Loss : Using __Cross Entropy__

                    - 相關概念：__Negative-Hard Mining__, __Score Minimum Threshold__, ...

                * Location Loss : Using __Smooth l1 Norm__

                    - 相關概念：__Match__ Ground Truth to Prior Boxes, __Jaccard Overlap__, ... 

            ### <u>_*✨ 其他細節部分可以參考：[__SSD論文__](https://arxiv.org/pdf/1512.02325.pdf) / [__教學-1__](https://medium.com/@bigwaterking01/ssd-single-shot-multibox-detector-%E4%BB%8B%E7%B4%B9-1fe95073c1a3) / [__教學-2__](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11)*_<u/>

<a name="reference"/>

## 參考資料

+ [__lzx1413/PytorchSSD__](https://github.com/lzx1413/PytorchSSD)

+ [__kumar-shridhar/PyTorch-BayesianCNN__](https://github.com/kumar-shridhar/PyTorch-BayesianCNN/tree/master/Image%20Recognition)

+ [__lufficc/SSD__](https://github.com/lufficc/SSD)

+ 其他 Pytorch [__官網資料__](https://pytorch.org/)、[__Github__](https://github.com/pytorch/pytorch)
