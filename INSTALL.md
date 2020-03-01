<a name="environment"/>

# 環境設置及安裝套件

##  基本環境：

- __python 3.6__ 或是以上
- __Linux 16.04 以上之環境__ 或是 __Linux-Like__ _的系統，如：__Mac__
- 需含 __pip__ 套件
- With GPU 1080 Ti / Or [Colab](https://colab.research.google.com/) from Google

## 環境設置：

- 若使用 Colab，需在指令前加入 “!”：

```shell
git clone https://github.com/bill0812/Detection-Tutorial.git
mv Detection-Tutorial/ detection_tutorial/
cd detection_tutorial
```

- 用 pip 安裝 pipenv

```shell
pip install pipenv
```

- (Optional) __用 brew 安裝 [pyenv](https://github.com/pyenv/pyenv) / [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)__，並且利用指令建造 __detection_tutorial__ 的虛擬環境，進入後再接下面步驟

- 用 pipenv 安裝 pipfile 裡面的環境，包括套件以及 python version ...，接著 __再啟動該虛擬環境__，若是用 Colab，則不需啟用，大多套件他都裝好了

```shell
pipenv install
pipenv shell
```

### ✨接著你就會擁有一個含有此 Repository 所需套件的一個獨立虛擬環境，包括套件：

+ opencv-python = " * "
+ numpy = " * "
+ pytorch = " * "