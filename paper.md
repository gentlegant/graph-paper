# 基于graphwave的网络embedding

1.  logbin的时间复杂度证明，理论暂时陷入了困境， 实验证明中
2.  0值的过滤，把diffusuion 矩阵变成稀疏矩阵，确实增加了效率
3.



项目github地址：

https://github.com/kkxhh/graphwave

摘要：
在图中较远的不同节点可能会具有相同的角色，对相同和相似角色的节点的识别可以服务不同的机器学习任务。该任务一般通过对节点embedding的相似性来完成。
网络嵌入是图学习的一个重要方向，不同的学习方法关注的视角不同，其目的都在于为图中的节点学习出能够捕捉重要feature的embedding，学习算法的快速和学得结果的高效是衡量此类算法性能的重要因素。  
本文提出了一种基于graphwave<sup>[1]</sup>的算法的改进，我们的模型通过两步为每个节点计算出一个低维embedding。同graphwave，我们通过特征分解和kernel方法快速的计算出了diffusion pattern。该diffusion pattern捕获了节点的拓扑信息，graphwave将diffusion pattern视作概率分布，采用特征函数来描绘概率分布，最终得到的embedding是特征函数的采样点。相较于graphwave，我们将diffusion pattern做了不同的处理来计算embedding，降低了embedding的维度，不仅如此，实验数据表明该方法在精确度上有了明显提升。下文中我们将展示在real world和标准拓扑中，我们同样的获得了state of art baseline。
## 模型简介

为了识别节点的角色信息，节点的周围拓扑至关很重要，越靠近的邻居对节点的拓扑影响越大，为了快速的捕捉节点的周围拓扑，我们使用了一种kernel方法，允许每个节点向周围邻居传送一定的能量，跳数越少的邻居能够得到的越多的能量。最终我们以对称矩阵的形式记录了任意节点对之间的能量传递情况。通过这样的方式，节点的拓扑信息被高效的capture在了我们的能量矩阵Ψ中，我们称该矩阵Ψ的每一行Ψ<sub>a</sub>为节点a的diffsuion pattern。然而，尽管diffusion 捕捉了节点的拓扑信息，我们却不能直接将该向量用于节点的相似性计算（如l<sub>2</sub>距离，minskov距离等） without specifying a one to one map between diffusion values，为了克服这个问题，一个intuitive 的做法是将diffsuion pattern排序然后比较排序后的diffsuion pattern。但排序带来的时间复杂度过高，且diffsuion pattern中存在大量的冷值容易影响计算结果。为了克服以上问题，我们提出了一种称为对数装箱的方法，该方法也是基于排序的，我们基于快速排序的思想快速将diffusion pattern分成多个箱子，如下图所示，数据在不同箱子间是有序的，但在箱内是无序的，箱子的宽度呈指数级下降，该方法的时间复杂度可以达到：O   。我们统计箱子的信息，最终生成了$\log_p n$维度的embedding。

![](./bins示意图.jpg '图1')





## 具体介绍


在不同领域，Kernels 在有效且快速的捕捉geometrical propertiesz中展现了优秀的能力<sup>[2，3，4]</sup>。kernels在计算得到diffusion pattern后还面临了解决one to one map和维度过高的问题。为了方便说明情况，我们图1为例（这个图后面肯定要改），该图中的节点共有五种角色。其各自的diffusion pattern如图二所示。在图二中，每一个图代表了一个节点的diffusion pattern,相同行的节点的角色是相同或者相近的。
![](./1.jpg '图1')
图1

![](./2.jpg '图2')   
图2
 ![](./3.jpg '图3')  
图3

如图2所示，尽管相同行的diffusion pattern来自相同角色的node，但其结果却是大相径庭的。为了衡量这样的diffusion pattern所表征的信息，一个启发式的做法是将diffusion排序，计算排序后的diffusion之间的相似性，在我们的实验代码中，我们在小规模数据上给出了排序做法的结果，其准确率是很高的，，但这样的做法会带来两个弊端：1是时间复杂度太大，2是结果的维度太高。graphwave的解决办法是将diffusion pattern看作概率分布，用特征函数来捕捉概率分布，最终得到的embedding是特征函数的采样点，我们指出了该做法的一些局限性：1部分采样点的信息不具有有效意义2该方法需要多次遍历diffusion pattern，（耗时大）3也是最重要的，该方法对于节点的拓扑信息丢失是严重的，由于特征函数的做法是独立于原本的概率分布的， 其没有考虑到diffusion本身的特征，（因此信息丢失严重）
为了解决以上提到的问题，同时弥补graphwave的不足，我们采用了一种称为对数垂直装箱的方法，该方法能在近似于线性的时间（需要复杂的概率论证明）内为每个节点generate 一个低维的embedding。   
（另外，如图2，3所示,diffusion中大量的冷值（接近于0）在计算中是没有意义的在考虑到这样的性质的情况下，我们可以可选地过冷值。（冷值的过滤是有意义的，能够极大的减少计算复杂度）

模型介绍

综上，我们提出的模型如下：
algothrim 1:

Input: Graph G = (V, E), scale s,p
Output: Structural embedding χa ∈ R2d for every node a ∈ V

compute the adjacent matrix A and diagonal degree matrix D of G
compute laplacian matrix L of G :L=D-A
对L矩阵作特征值分解，得到特征矩阵U和特征值矩阵Λ。
compute Ψ=U g<sub>s</sub>(Λ) *U<sup>T</sup>
Ψ中的第m行 表示了节点m的diffsuion pattern。
对于Ψ的每一行，执行垂直对数装箱，生成一个更低维的向量

装箱操作的伪代码如下：
def bins(array,p):
    res=[]
    while array不为空:
        选取array中最小的p（比例）个数，将这p个数的均值加入res
        从array中删除以上p个数
    return res


由此我们将原本的n维向量变成了$\log_p n$维度。

output:对于每个节点a，生成bin<sub>a</sub>


##  contribution

为了解决diffusion pattern之间one to one map不明确的问题，我们提出了一种称为对数垂直装箱的模型，其运算速度和精度在实验中都表现出了高于graphwave的性能。


## experiment



为了说明我们提出算法的优越性，我们在标准拓扑和来自real-world的数据集分别进行了实验，其结果如下：
### experiment on synthetic graphs

#### barbel graph
 ![](./barbelresult.png '图4')










#### A cycle graph with attached “house” shapes
s=1
参数：add_edge=5
n_shapes = 5 

 ![](./4.jpg '图4')
 一个标准拓扑
 ![](./5.jpg '图5')
 diffusion pattern 
 ![](./6.jpg '图6')
 graphwave生成的embedding  100维

 ![](./7.jpg '图7')

 我们的方法生成的embedding  6维
 ![](./8.jpg '图8')
 耗时和准确率