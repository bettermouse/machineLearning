#### logistic regression


- Ŀ��
```math
Wx+b=0  
```
- sigmod
```math
f(x)=\frac{1}{1+e^{-x}}
```
- sigmod��    

```math
f\prime(x)=\frac{e^{-x}}{(1+e^{-x})^{2}}=f(x)[1-f(x)]
```
- sigmodʵ��
``` python
def sig(x):
    '''Sigmoid����
    input:  x(mat):feature * w
    output: sigmoid(x)(mat):Sigmoidֵ
    '''
    return 1.0 / (1 + np.exp(-x))
```
- ����x���������ĸ���
```math
P(y=1|X,W,b)=\sigma(WX+b)=\frac{1}{1+e^{-{(WX+b)}}}
```
- ����x���ڸ����ĸ���
```math
P(y=1|X,W,b)=1-\sigma(WX+b)=\frac{e^{-{(WX+b)}}}{1+e^{-{(WX+b)}}}
```
##### ��ʧ����
- X�������y�ĸ���

```math
P(y|X,W,b)=\sigma(WX+b)^{y}(1-\sigma(WX+b))^{1-y}=
```
- ������Ȼ��,��Ȼ����

```math
L_{W,b}=\prod_{i=0}^m (h_{W,b}(X^{i}))^{y^{i}}(1-h_{W,b}(X^{i}))^{1-y^{i}}

h_{W,b}(X^{i})=\sigma(WX^{i}+b)
```
- ����log,����





\frac{}{}{}{7x+5}{1+y^2}

