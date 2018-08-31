## Pytoch_DANN
> This is a implementation of [Domain-Adversarial Training of Neural Networks][1]  
> with pytorch. This paper introduced a simple and effective method for accompli-  
> shing domian adaptation with SGD with a GRL(Gradient Reveral Layer). According   
> to this paper, domain classifier is used to decrease the H-divergence between  
> source domain distribution and target domain distribution. For the tensorflow  
> version, you can see [tf-dann][2].

### requirements
> python3.6.2  
> `pip install -r requirements.txt`

### Data
> In this work, MNIST and MNIST_M datasets are used in experiments. MNIST dataset  
> can be downloaded with `torchvision.datasets`. MINIST_M dataset can be downloa-  
> ded at [Yaroslav Ganin's homepage][3]. Then you can extract the file to your data dire-  
> ctory and run the `preprocess.py` to make the directory able to be used with  
> `torchvision.datasets.ImageFolder`:
```
python preprocess.py
```

### Experiments
> You can run `main.py` to implements the MNSIT experiments for the paper with the  
> similar model and same paramenters.The paper's results and this work's results a-  
> re as follows:  

|Method     | Target Acc(paper) | Target Acc(this work)|
|:----------:|:-----------------:|:---------------------:|
|Source Only| 0.5225            | 0.5189|
|DANN       | 0.7666            | 0.7600|``````

> Experiment on SVHN->MNIST is added in this project, but some bugs are not fixed.  
> The accuracies of source and target domains are not good at the same time.



[1]:https://arxiv.org/pdf/1505.07818.pdf
[2]:https://github.com/pumpikano/tf-dann
[3]:http://yaroslav.ganin.net/
