#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件分类器
实现了一个基础的邮件分类器，支持多种特征提取方法
"""

import re
import os
import numpy as np
from jieba import cut
from itertools import chain
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


class EmailClassifier:
    """
    邮件分类器类
    支持高频词特征和TF-IDF特征两种特征提取方式
    """
    
    def __init__(self, feature_method='freq', top_words_num=100):
        """
        初始化分类器
        
        Args:
            feature_method (str): 特征提取方法，'freq'表示高频词特征，'tfidf'表示TF-IDF特征
            top_words_num (int): 高频词特征时选择的词汇数量
        """
        self.feature_method = feature_method
        self.top_words_num = top_words_num
        self.model = MultinomialNB()
        self.top_words = []
        self.tfidf_vectorizer = None
        self.all_documents = []
        
    def preprocess_text(self, filename):
        """
        文本预处理函数
        
        Args:
            filename (str): 文件路径
            
        Returns:
            list: 预处理后的词列表
        """
        words = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                # 过滤无效字符
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                # 使用jieba.cut()方法对文本切词处理
                line = cut(line)
                # 过滤长度为1的词
                line = filter(lambda word: len(word) > 1, line)
                words.extend(line)
        return words
    
    def get_document_text(self, filename):
        """
        获取文档的完整文本（用于TF-IDF）
        
        Args:
            filename (str): 文件路径
            
        Returns:
            str: 处理后的文本字符串
        """
        words = self.preprocess_text(filename)
        return ' '.join(words)
    
    def build_freq_features(self, file_list):
        """
        构建高频词特征
        
        Args:
            file_list (list): 文件列表
            
        Returns:
            tuple: (特征矩阵, 高频词列表)
        """
        all_words = []
        
        # 提取所有文档的词
        for filename in file_list:
            words = self.preprocess_text(filename)
            all_words.append(words)
        
        # 统计词频并获取高频词
        freq = Counter(chain(*all_words))
        self.top_words = [word[0] for word in freq.most_common(self.top_words_num)]
        
        # 构建特征向量
        vector = []
        for words in all_words:
            word_map = [words.count(word) for word in self.top_words]
            vector.append(word_map)
        
        return np.array(vector), self.top_words
    
    def build_tfidf_features(self, file_list):
        """
        构建TF-IDF特征
        
        Args:
            file_list (list): 文件列表
            
        Returns:
            tuple: (特征矩阵, TF-IDF向量化器)
        """
        # 获取所有文档的文本
        documents = []
        for filename in file_list:
            text = self.get_document_text(filename)
            documents.append(text)
            
        self.all_documents = documents
        
        # 使用TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.top_words_num,
            stop_words=None,
            token_pattern=r'\b\w+\b'
        )
        
        # 拟合并转换文档
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        return tfidf_matrix.toarray(), self.tfidf_vectorizer
    
    def train(self, data_dir='邮件_files', spam_range=(0, 127), ham_range=(127, 151)):
        """
        训练分类器
        
        Args:
            data_dir (str): 数据目录
            spam_range (tuple): 垃圾邮件文件编号范围
            ham_range (tuple): 正常邮件文件编号范围
        """
        # 构建文件列表
        file_list = []
        labels = []
        
        # 添加垃圾邮件
        for i in range(spam_range[0], spam_range[1]):
            file_path = os.path.join(data_dir, f'{i}.txt')
            if os.path.exists(file_path):
                file_list.append(file_path)
                labels.append(1)  # 垃圾邮件标记为1
        
        # 添加正常邮件
        for i in range(ham_range[0], ham_range[1]):
            file_path = os.path.join(data_dir, f'{i}.txt')
            if os.path.exists(file_path):
                file_list.append(file_path)
                labels.append(0)  # 正常邮件标记为0
        
        print(f"加载了 {len(file_list)} 个邮件文件")
        print(f"垃圾邮件: {sum(labels)} 个, 正常邮件: {len(labels) - sum(labels)} 个")
        
        # 根据特征方法构建特征
        if self.feature_method == 'freq':
            print("使用高频词特征提取方法")
            features, _ = self.build_freq_features(file_list)
            print(f"提取了 {len(self.top_words)} 个高频词作为特征")
            print("前10个高频词:", self.top_words[:10])
        elif self.feature_method == 'tfidf':
            print("使用TF-IDF特征提取方法")
            features, _ = self.build_tfidf_features(file_list)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            print(f"提取了 {len(feature_names)} 个TF-IDF特征")
            print("前10个特征词:", feature_names[:10].tolist())
        else:
            raise ValueError("feature_method必须是'freq'或'tfidf'")
        
        # 训练模型
        self.model.fit(features, labels)
        print(f"特征矩阵形状: {features.shape}")
        print("模型训练完成!")
        
        return features, labels
    
    def predict(self, filename):
        """
        预测单个邮件的分类
        
        Args:
            filename (str): 邮件文件路径
            
        Returns:
            str: 分类结果
        """
        if self.feature_method == 'freq':
            # 使用高频词特征
            words = self.preprocess_text(filename)
            current_vector = np.array([words.count(word) for word in self.top_words])
            current_vector = current_vector.reshape(1, -1)
        elif self.feature_method == 'tfidf':
            # 使用TF-IDF特征
            text = self.get_document_text(filename)
            current_vector = self.tfidf_vectorizer.transform([text]).toarray()
        else:
            raise ValueError("feature_method必须是'freq'或'tfidf'")
        
        # 预测
        result = self.model.predict(current_vector)
        probability = self.model.predict_proba(current_vector)
        
        label = '垃圾邮件' if result[0] == 1 else '正常邮件'
        confidence = max(probability[0])
        
        return label, confidence
    
    def evaluate(self, test_files):
        """
        评估模型性能
        
        Args:
            test_files (list): 测试文件列表，格式为[(filename, true_label), ...]
            
        Returns:
            dict: 评估结果
        """
        predictions = []
        true_labels = []
        
        for filename, true_label in test_files:
            try:
                pred_label, confidence = self.predict(filename)
                pred_numeric = 1 if pred_label == '垃圾邮件' else 0
                predictions.append(pred_numeric)
                true_labels.append(true_label)
                print(f"{os.path.basename(filename)}: 预测={pred_label}, 置信度={confidence:.3f}")
            except Exception as e:
                print(f"预测 {filename} 时出错: {e}")
        
        if predictions:
            # 计算准确率
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
            print(f"\n准确率: {accuracy:.3f}")
            
            # 打印详细报告
            print("\n分类报告:")
            print(classification_report(true_labels, predictions, 
                                      target_names=['正常邮件', '垃圾邮件']))
            
            print("\n混淆矩阵:")
            print(confusion_matrix(true_labels, predictions))
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_labels': true_labels
            }
        
        return {}


def main():
    """
    主函数，演示两种特征提取方法的使用
    """
    print("=== 邮件分类器测试 ===\n")
    
    # 测试文件（假设这些文件存在）
    test_files = [
        ('邮件_files/151.txt', 1),  # 假设是垃圾邮件
        ('邮件_files/152.txt', 1),
        ('邮件_files/153.txt', 1),
        ('邮件_files/154.txt', 1),
        ('邮件_files/155.txt', 0),  # 假设是正常邮件
    ]
    
    # 测试高频词特征
    print("1. 测试高频词特征方法")
    print("-" * 50)
    classifier_freq = EmailClassifier(feature_method='freq', top_words_num=100)
    classifier_freq.train()
    
    print("\n测试邮件分类结果:")
    for filename, _ in test_files:
        try:
            result, confidence = classifier_freq.predict(filename)
            print(f"{os.path.basename(filename)}: {result} (置信度: {confidence:.3f})")
        except FileNotFoundError:
            print(f"{os.path.basename(filename)}: 文件不存在")
    
    print("\n" + "="*60 + "\n")
    
    # 测试TF-IDF特征
    print("2. 测试TF-IDF特征方法")
    print("-" * 50)
    classifier_tfidf = EmailClassifier(feature_method='tfidf', top_words_num=100)
    classifier_tfidf.train()
    
    print("\n测试邮件分类结果:")
    for filename, _ in test_files:
        try:
            result, confidence = classifier_tfidf.predict(filename)
            print(f"{os.path.basename(filename)}: {result} (置信度: {confidence:.3f})")
        except FileNotFoundError:
            print(f"{os.path.basename(filename)}: 文件不存在")
    
    print("\n" + "="*60)
    print("测试完成!")


if __name__ == "__main__":
    main() 