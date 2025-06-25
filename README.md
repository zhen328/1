# NLP项目：邮件分类器

## 项目概述

本项目实现了一个基于机器学习的邮件分类器，用于识别垃圾邮件和正常邮件。项目支持多种特征提取方法，包括高频词特征和TF-IDF特征，并采用多项式朴素贝叶斯分类器进行分类。

## 核心功能

- **邮件文本预处理**：支持中文文本分词、停用词过滤、特殊字符清理
- **多种特征提取方式**：
  - 高频词特征：基于词频统计的传统方法
  - TF-IDF特征：考虑词汇重要性的加权方法
- **朴素贝叶斯分类**：基于概率模型的高效分类算法
- **模型评估**：提供准确率、分类报告、混淆矩阵等评估指标
- **参数化配置**：支持特征方法的灵活切换

## 算法基础

### 多项式朴素贝叶斯分类器

本项目采用多项式朴素贝叶斯(Multinomial Naive Bayes)分类器，其核心基于以下数学原理：

#### 贝叶斯定理
```
P(类别|特征) = P(特征|类别) × P(类别) / P(特征)
```

#### 特征独立性假设
朴素贝叶斯假设各特征之间相互独立，即：
```
P(x₁,x₂,...,xₙ|类别) = P(x₁|类别) × P(x₂|类别) × ... × P(xₙ|类别)
```

#### 在邮件分类中的应用
对于邮件分类任务，我们需要计算：
- **P(垃圾邮件|邮件特征)**：给定邮件特征下是垃圾邮件的概率
- **P(正常邮件|邮件特征)**：给定邮件特征下是正常邮件的概率

多项式朴素贝叶斯特别适合处理文本数据中的词频特征，因为：
1. 它假设特征值符合多项式分布
2. 能够有效处理稀疏的高维特征空间
3. 对小样本数据具有良好的泛化能力

## 数据处理流程

### 1. 文本预处理
```python
def preprocess_text(filename):
    # 读取文件内容
    # 过滤特殊字符：[.【】0-9、——。，！~\*]
    # 使用jieba进行中文分词
    # 过滤长度为1的词（去除单字符）
    return words
```

**处理步骤**：
- **字符清理**：移除标点符号、数字、特殊符号
- **中文分词**：使用jieba库进行精确分词
- **词长过滤**：保留长度大于1的词，提高特征质量
- **编码处理**：统一使用UTF-8编码确保中文正确处理

### 2. 停用词处理
虽然当前实现中未显式使用停用词表，但通过以下方式实现了停用词的间接过滤：
- 特殊字符过滤去除了大部分标点类停用词
- 单字符过滤去除了部分高频但无意义的词
- 高频词/TF-IDF方法会自动调节常见词的权重

## 特征构建过程

### 1. 高频词特征选择

#### 数学表达
高频词特征基于词频统计：
```
特征向量 = [count(word₁), count(word₂), ..., count(wordₙ)]
其中：count(wordᵢ) 表示词 wordᵢ 在文档中出现的次数
```

#### 实现逻辑
```python
def build_freq_features(file_list):
    # 1. 统计所有文档的词频
    freq = Counter(chain(*all_words))
    # 2. 选择出现频率最高的N个词
    top_words = [word[0] for word in freq.most_common(N)]
    # 3. 构建特征向量
    vector = [[words.count(word) for word in top_words] for words in all_words]
    return vector
```

**特点**：
- **简单直观**：直接统计词频，计算复杂度低
- **全局视角**：基于整个语料库的词频分布
- **适用性强**：适合词汇分布相对均匀的文本

### 2. TF-IDF特征加权

#### 数学表达
TF-IDF(Term Frequency-Inverse Document Frequency)结合了词频和逆文档频率：

```
TF-IDF(t,d) = TF(t,d) × IDF(t)

其中：
TF(t,d) = count(t,d) / |d|  # 词t在文档d中的频率
IDF(t) = log(|D| / |{d ∈ D : t ∈ d}|)  # 逆文档频率

|d|: 文档d的总词数
|D|: 总文档数
|{d ∈ D : t ∈ d}|: 包含词t的文档数
```

#### 实现逻辑
```python
def build_tfidf_features(file_list):
    # 1. 获取文档文本
    documents = [get_document_text(filename) for filename in file_list]
    # 2. 使用TfidfVectorizer计算TF-IDF值
    tfidf_vectorizer = TfidfVectorizer(max_features=N)
    # 3. 拟合并转换文档
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_matrix.toarray()
```

**特点**：
- **权重平衡**：TF体现词的重要性，IDF降低常见词权重
- **信息量导向**：突出稀有但有区分度的词汇
- **标准化处理**：自动处理文档长度差异的影响

### 3. 两种方法的对比

| 特征方法 | 数学基础 | 优势 | 劣势 | 适用场景 |
|---------|---------|------|------|---------|
| **高频词特征** | 简单词频统计 | 计算简单、直观易懂 | 忽略词汇重要性差异 | 词汇分布均匀的文本 |
| **TF-IDF特征** | 频率×逆文档频率 | 考虑词汇区分度、信息量大 | 计算复杂、需要全局统计 | 词汇重要性差异大的文本 |

## 特征模式切换方法

### 使用方式

#### 1. 导入模块
```python
from email_classifier import EmailClassifier
```

#### 2. 初始化分类器
```python
# 使用高频词特征
classifier_freq = EmailClassifier(feature_method='freq', top_words_num=100)

# 使用TF-IDF特征  
classifier_tfidf = EmailClassifier(feature_method='tfidf', top_words_num=100)
```

#### 3. 训练和预测
```python
# 训练模型
classifier.train(data_dir='邮件_files')

# 预测单个邮件
result, confidence = classifier.predict('test_email.txt')
print(f"分类结果: {result}, 置信度: {confidence:.3f}")
```

### 参数说明

- **feature_method**: 
  - `'freq'`: 使用高频词特征
  - `'tfidf'`: 使用TF-IDF特征
- **top_words_num**: 特征词汇数量，默认100个
- **data_dir**: 训练数据目录路径
- **spam_range**: 垃圾邮件文件编号范围
- **ham_range**: 正常邮件文件编号范围

## 项目结构

```
NLP/
├── README.md                    # 项目文档
├── Classification/              # 邮件分类模块
│   ├── README.md               # 分类器说明
│   ├── classify.ipynb          # Jupyter笔记本实现
│   ├── email_classifier.py     # Python模块实现
│   ├── 邮件_files/             # 邮件数据
│   │   ├── 0.txt ~ 126.txt    # 垃圾邮件
│   │   └── 127.txt ~ 150.txt  # 正常邮件
│   └── image/                  # 图片资源
└── jieba/                      # 中文分词模块
    ├── main_jieba.ipynb
    ├── dictionary_based_segmentation.ipynb
    └── userdict.txt
```

## 环境要求

### Python版本
- Python 3.7+

### 依赖包
```bash
pip install jieba scikit-learn numpy pandas matplotlib
```

### 详细依赖
- **jieba**: 中文分词
- **scikit-learn**: 机器学习算法
- **numpy**: 数值计算
- **pandas**: 数据处理（可选）
- **matplotlib**: 可视化（可选）

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd NLP
```

### 2. 安装依赖
```bash
pip install -r requirements.txt  # 如果有requirements.txt
# 或者手动安装
pip install jieba scikit-learn numpy
```

### 3. 运行示例
```bash
cd Classification
python email_classifier.py
```

### 4. 在Jupyter中运行
```bash
jupyter notebook classify.ipynb
```

## 使用示例

### 基础使用
```python
from email_classifier import EmailClassifier

# 创建分类器实例
classifier = EmailClassifier(feature_method='tfidf', top_words_num=100)

# 训练模型
classifier.train()

# 预测新邮件
result, confidence = classifier.predict('new_email.txt')
print(f"预测结果: {result}, 置信度: {confidence:.3f}")
```

### 批量评估
```python
# 准备测试数据
test_files = [
    ('test1.txt', 1),  # (文件路径, 真实标签: 1=垃圾邮件, 0=正常邮件)
    ('test2.txt', 0),
]

# 评估性能
results = classifier.evaluate(test_files)
print(f"准确率: {results['accuracy']:.3f}")
```

## 性能优化建议

1. **数据预处理优化**：
   - 构建专用的中文停用词表
   - 添加同义词合并功能
   - 实现词干化/词元化

2. **特征工程改进**：
   - 尝试N-gram特征
   - 添加文本长度、特殊字符比例等统计特征
   - 考虑Word2Vec等词向量方法

3. **模型改进**：
   - 尝试其他分类算法（SVM、随机森林等）
   - 使用集成学习方法
   - 添加交叉验证评估

4. **系统优化**：
   - 实现模型持久化（保存/加载）
   - 添加增量学习功能
   - 优化大文件处理性能

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至: [your-email@example.com]

---

*最后更新: 2024年*


![皮卡丘](https://github.com/user-attachments/assets/d1a6bcc5-3ef5-46b8-b7ab-46c958954177)

## 基础档案 
可爱的小不点电气鼠
### 外貌特征 
- 黄色的
- 大大的眼睛
- 静电
- 耳朵尖端黑色

### 我的朋友
1. 杰尼龟
2. 胖丁
3.小智 


### 重要坐标
- 🏠 **住址**: [皮卡丘之家] 
- 🏢 **工作单位**: [....]

### 日常作息表
| 时间       | 事项                  |
|------------|-----------------------|
| 7:00 AM    |     皮一下   |
| 12:00 AM   |      吃训练     |
| 4:00 PM    |    |玩游戏
| 19:00 PM   |        学   |

### 人生信条
> ""
---世上无难事只怕有心人

## 我的专业是人工智能
### 我最喜欢的一段代码
def keep_fighting():
    life = "hard"
    success = False
    failures = 0
    
    while not success:
        try:
            print("Keep coding...")
            failures += 1
            if failures > 99:  # 失败是成功之母
                success = True
        except Exception as e:
            print(f"Bug found: {e}. Debugging...")  # 遇到问题就解决它
        finally:
            print("Never give up!\n")
    
    print(">>>> SUCCESS! <<<<")  # 终会成功




### 我最喜欢的环境管理工具是conda
![2](https://github.com/user-attachments/assets/a4e96af2-4a86-4ce9-8dad-67f64bd49faf)


### 我可以在IDE上使用我建立的虚拟环境
![3](https://github.com/user-attachments/assets/f043fc20-ccba-4fc3-9c0e-7368b6bdb7fe)


