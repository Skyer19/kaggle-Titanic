# kaggle-Titanic

## 数据处理时遇到的苦难
在机器学习算法中，我们经常会遇到分类特征，例如：人的性别有男女等。这些特征值并不是连续的，而是离散的，无序的。通常我们需要对其进行特征数字化。<br>
<br>
**解决方法：独热编码**

## 独热编码（One-Hot）
One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。
One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都是零值，它被标记为1。

