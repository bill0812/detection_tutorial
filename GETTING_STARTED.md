<a name="guide"/>

# 快速教學指南

### 👉 這個 Markdown 說明了此 Repository 的 __「 檔案架構 」__，同時也會說明如何利用這些檔案來 __「 完成訓練及測試 」__，或是 __「 其他功能用途 」__

<a name="directory_tree"/>

## 了解目錄樹

```shell
Detection_Tutorial
├── Apis
│   ├── config.py
│   ├── __init__.py
│   ├── test.py
│   └── train.py
├── Config
│   └── ssd_config.py
├── Datasets
│   ├── augmentation.py
│   ├── custom.py
│   └── __init__.py
├── __init__.py
├── main.py
├── Model
│   ├── Anchor_Head
│   │   └── ssd_head.py
│   ├── Backbone
│   │   ├── __init__.py
│   │   └── ssd_vgg.py
│   ├── Detector
│   │   ├── __init__.py
│   │   └── single_stage_detector.py
│   ├── __init__.py
│   ├── Neck
│       └── __init__.py
├── Pipfile
├── Pipfile.lock
└── Utils
    ├── box_utils.py
    ├── checkpoint.py
    ├── detect.py
    ├── __init__.py
    ├── l2norm.py
    └── loss.py
```

- __Apis__ :

    + 包括 main.py 所需要的檔案，如 train.py / test.py，以及獲取訓練參數、測試參數...等各式參數的 config.py
    
    + 此一 config.py 為進入 train.py / test.py 前的手續動作

- __Config__ :

    + 包括訓練、測試、資料集、模型參數的 config file，我們利用此以檔案餵入 main.py 來完成整個過程

- __Datasets__ :

    + 包括 custom.py，此一檔案負責客製化各種資料集，供  dataloader 使用

    + 也包括 augmentation.py，負責對餵進模型的資料進行前處理 -- __「 資料擴增 」__

- __Model__ :

    + 主要包括 __backbone / anchor head / detector__，其中 detector 為結合 backbone / anchor head ...，為建模型的 main file。以下將主要介紹 __backbone / anchor head__

    + __backbone__：存放模型的骨架，意思就是裡面的 Net 為模型的到各層 feature maps 的不同架構

    + __anchor head__：此一部分會生成所要的 anchor boxes，同時將各層 feature maps 利用此 head 來預測出我們所要的結果

- __Utils__ :

    + 包含各式功能的函式、物件...，如 Loss function / 處理 bounding boxes / detection from model's result

<a name="train_test"/>

## 如何訓練以及測試

### 👉 在這邊我們提供一個 Overview 的 Colab 檔案讓大家簡單瞭解 SSD 怎麼訓練及測試；也提供一個 Detail 版本的 main.py 讓大家可以仔細了解：

    1. 引用了哪些套件
    2. 如何 Load 資料
    3. Training / Validation / Testing  的過程

- Overview 版本：[overview.ipynb](overview_main.ipynb)

- Detail 版本：[detail.ipynb](detail_main.ipynb)

    - 若要用 [main.py](main.py)，以 __「指令」__ 的方式來訓練，則需：
    
    ```shell
    python main.py --config_file Config/ssd_config.py
    ```

    - 若要自己建構模型模組，可以根據、利用已存在的模組，調整 [Config](Config/) 裡面檔案的設定，__「 按照檔案格式調整架構 」__ 即可！

    ### ✨ 因為我們將模型模組化，因此只要調整在 Config 檔裡面調整模組，並確認模組存在，即可完成訓練以及測試，不需要親自更改模型中模組的程式碼，方便開發人員作業

<a name="config"/>

## 基本設定檔

- Example : [SSD_Config.py](Config/ssd_config.py)

    ```python
    ssd_model = {
        "input_size" : 300
        "backbone" : ...,
        "anchor_head" : ...,
        "prior_box" : ...,
        "loss_function" : ...
    }
    training_config = {
        "batch_size" : 32,
        "number_workers" : 4,
        "epoch" : 50,
        "learning_rate" : 0.001,
        "weight_decay" : 0.0005,
        "gamma" : 0.1,
        "gradient_clip" : True,
        "optimizer" : "adam",
        ...
    }
    testing_config = {
        "config" : ...,
        "checkpoint" : ...,
        ...
    }
    dataset = {
        "dataset_name" : "test",
        "training_data" : ...,
        ...
        "dataloader" : ...
    }
    ```
    以上為示意，詳細 __「 Dictionary 」__ 請看檔案

- 檔案介紹：主要包括

    + __ssd_model__：

        => 包括 __backbone / anchor_head / loss_function__ 等，這些基本架構之參數都是在創建模型時所需

    + __training_config：__

        => 包括 __batch_size / number_workers / optimizer's 參數__ ...，這些都是訓練過程中所需要的參數，一併在此定義好

    + __testing_config：__

        => 包括 __checkpoint / config__ 參數 ...，這些都是測試過程中所需要的參數，相對訓練參數較少

    + __dataset：__

        => 包括 __dataloader 所需的標籤檔案 / 訓練、驗證或是測試時所需的 annotation 檔案 / data augmentation 所需的相關變數__ ...