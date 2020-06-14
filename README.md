# Pytorch Create Network

2020.6

Gao Junbin , Northeastern University at Qinhuangdao , Hebei , China

---

## Introduction

I want to build a common template (based on Pytorch) to implement the common network model in deep learning.

I will continue to improve the content of the template in the future, and try my best to gradually add some network structure models.

I will updata some details on my [gitbook](https://junbin.gitbook.io/studynotes/).

I am a novice, welcome to criticize and guide.

## Requirement

- Ubuntu > 14.04 / Windows > 7 / Macos > 10.14
- Python 3 (recommend Anaconda3)
- Pytorch 1.5.0

## Train

```shell
python Train.py --config './Config/Config.yaml'
```

You can modify params in `Config/Config.yaml`. And you may need to modify the dataset reload code for fit your own project and datasets.

Suggest you to organize files in the following structure:

> Train.py
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







