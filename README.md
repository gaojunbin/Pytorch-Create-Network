# Pytorch Create Network

2020.6

Gao Junbin , Northeastern University at Qinhuangdao , Hebei , China

Visit count : [![HitCount](http://hits.dwyl.com/gaojunbin/Pytorch-Create-Network.svg)](http://hits.dwyl.com/gaojunbin/Pytorch-Create-Network)

About Author : [![](https://badgen.net/badge/icon/Website?icon=chrome&label)](http://junbin.xyz) 

---

## 目录

- [Introduction](#Introduction)
- [Requirement](#Requirement)
- [Train](#Train)
- [Inference](#Inference)
- [Clear](#Clear)

## Introduction

I want to build a common template (based on Pytorch) to implement the common network model in deep learning.

I will continue to improve the content of the template in the future, and try my best to gradually add some network structure models.

I will updata some details on my [gitbook](https://junbin.gitbook.io/studynotes/).

I am a novice, welcome to criticize and guide.

## Requirement 

- Ubuntu >= 16.04 / Macos >= 10.14
- Python 3 (recommend Anaconda3)
- Pytorch 1.5.0

Attention: Theoretically, you can also run on windows>=8, except the shell files.

## Train

If this is your first cloning, you should install the enviroment mentioned in [Requirement](#Requirement).

Then run the following command，

```shell
sh Train.sh
```

You can modify params in `Config/Config.yaml`. And you may need to modify the dataset reload code for fit your own project and datasets.

Suggest you to organize files in the following structure:

> Train.py
>
> Train.sh
>
> Inference.py
>
> Clear.sh
>
> Config
>
> > Config.yaml
>
> Network
>
> > Network.py
> >
> > ...
>
> Data
>
> > DataReload.py
> >
> > DataPre.py
> >
> > Dataset
> >
> > > class1
> > >
> > > class2
> > >
> > > ...
>
> ...

## Inference

You can modify the ```Inference.py```to use the model after training to inference your data. Of cource, you can add some your own codes to complete the batch inference and realize visualization function.

```shell
python Inference.py --config ./Config/Config.yaml
```

## Clear

If you wat to make the project to clear the cache during you run the codes or train the models. You can run the following command，

```shell
sh Clear.sh
```

Attention: This command will delet all the modles and logs without prompting! Please careful operation. You are advised to make a backup if necessary.