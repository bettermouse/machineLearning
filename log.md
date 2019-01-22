# 机器学习-如何在github上写数学公式
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
居中格式: $$xxx$$
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
靠左格式: \\(xxx\\)
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)
测试
$$\frac{7x+5}{1+y^2}$$
\\(l(x_i) = - \log_2 P(x_i)\\)

#### logistic regression


- 目标
```math
Wx+b=0  
```
- sigmod
```math
f(x)=\frac{1}{1+e^{-x}}
```
- sigmod求导    

```math
f\prime(x)=\frac{e^{-x}}{(1+e^{-x})^{2}}=f(x)[1-f(x)]
```
- sigmod实现
``` python
def sig(x):
    '''Sigmoid函数
    input:  x(mat):feature * w
    output: sigmoid(x)(mat):Sigmoid值
    '''
    return 1.0 / (1 + np.exp(-x))
```
- 输入x属于正例的概率
```math
P(y=1|X,W,b)=\sigma(WX+b)=\frac{1}{1+e^{-{(WX+b)}}}
```
- 输入x属于负例的概率
```math
P(y=1|X,W,b)=1-\sigma(WX+b)=\frac{e^{-{(WX+b)}}}{1+e^{-{(WX+b)}}}
```
##### 损失函数
- X属于类别y的概率

```math
P(y|X,W,b)=\sigma(WX+b)^{y}(1-\sigma(WX+b))^{1-y}=
```
- 极大似然法,似然函数

```math
L_{W,b}=\prod_{i=0}^m (h_{W,b}(X^{i}))^{y^{i}}(1-h_{W,b}(X^{i}))^{1-y^{i}}

h_{W,b}(X^{i})=\sigma(WX^{i}+b)
```
- 加上log,负号





\frac{}{}{}{7x+5}{1+y^2}

