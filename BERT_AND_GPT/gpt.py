#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GPT的文本续写任务
使用Hugging Face的中文GPT模型进行文本续写
"""

import torch
from transformers import (
    GPT2LMHeadModel, 
    BertTokenizer,  # 用于中文GPT模型
    pipeline,
    set_seed
)
import warnings
warnings.filterwarnings('ignore')

class ChineseGPTGenerator:
    """中文GPT文本生成器"""
    
    def __init__(self, model_name="uer/gpt2-chinese-cluecorpussmall"):
        """
        初始化GPT模型和分词器
        
        Args:
            model_name (str): 预训练模型名称
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        print(f"正在加载模型: {model_name}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用pipeline方式加载...")
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_pipeline = True
            print("Pipeline加载成功！")
            return
        
        self.use_pipeline = False
    
    def generate_text_manual(self, prompt, max_length=200, temperature=0.8, 
                           top_p=0.9, repetition_penalty=1.2, num_return_sequences=1):
        """
        手动生成文本
        
        Args:
            prompt (str): 输入提示文本
            max_length (int): 最大生成长度
            temperature (float): 温度参数，控制随机性
            top_p (float): nucleus sampling参数
            repetition_penalty (float): 重复惩罚
            num_return_sequences (int): 返回序列数量
            
        Returns:
            list: 生成的文本列表
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 设置随机种子以确保可重复性
        set_seed(42)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.sep_token_id
            )
        
        # 解码生成的文本
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # 移除原始提示，只保留生成的部分
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def generate_text_pipeline(self, prompt, max_length=200, temperature=0.8, 
                             top_p=0.9, repetition_penalty=1.2, num_return_sequences=1):
        """
        使用pipeline生成文本
        
        Args:
            prompt (str): 输入提示文本
            max_length (int): 最大生成长度
            temperature (float): 温度参数
            top_p (float): nucleus sampling参数
            repetition_penalty (float): 重复惩罚
            num_return_sequences (int): 返回序列数量
            
        Returns:
            list: 生成的文本列表
        """
        try:
            results = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True
            )
            
            generated_texts = []
            for result in results:
                generated_text = result['generated_text']
                # 移除原始提示，只保留生成的部分
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                generated_texts.append(generated_text.strip())
            
            return generated_texts
        except Exception as e:
            print(f"Pipeline生成失败: {e}")
            return [f"生成失败: {str(e)}"]
    
    def generate_text(self, prompt, max_length=200, temperature=0.8, 
                     top_p=0.9, repetition_penalty=1.2, num_return_sequences=1):
        """
        生成文本的统一接口
        
        Args:
            prompt (str): 输入提示文本
            max_length (int): 最大生成长度
            temperature (float): 温度参数
            top_p (float): nucleus sampling参数
            repetition_penalty (float): 重复惩罚
            num_return_sequences (int): 返回序列数量
            
        Returns:
            list: 生成的文本列表
        """
        if self.use_pipeline:
            return self.generate_text_pipeline(
                prompt, max_length, temperature, top_p, repetition_penalty, num_return_sequences
            )
        else:
            return self.generate_text_manual(
                prompt, max_length, temperature, top_p, repetition_penalty, num_return_sequences
            )

def main():
    """主函数"""
    # 句子开头列表（根据学号倒数第一位选择）
    sentence_beginnings = [
        "如果我拥有一台时间机器",      # 0
        "当人类第一次踏上火星",        # 1
        "如果动物会说话，它们最想告诉人类的是",  # 2
        "有一天，城市突然停电了",      # 3
        "当我醒来，发现自己变成了一本书",  # 4
        "假如我能隐身一天，我会",      # 5
        "我走进了那扇从未打开过的门",    # 6
        "在一个没有网络的世界里",      # 7
        "如果世界上只剩下我一个人",      # 8
        "梦中醒来，一切都变了模样"       # 9
    ]
    
    # 学号：202210179070，倒数第一位是0
    student_id = "202210179070"
    last_digit = int(student_id[-1])
    selected_prompt = sentence_beginnings[last_digit]
    
    print("=" * 60)
    print("基于GPT的中文文本续写任务")
    print("=" * 60)
    print(f"学号: {student_id}")
    print(f"倒数第一位: {last_digit}")
    print(f"选择的句子开头: {selected_prompt}")
    print("=" * 60)
    
    try:
        # 初始化GPT生成器
        generator = ChineseGPTGenerator()
        
        print(f"\n开始生成文本，提示词: '{selected_prompt}'")
        print("-" * 60)
        
        # 生成多个版本的续写
        generated_texts = generator.generate_text(
            prompt=selected_prompt,
            max_length=150,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=3
        )
        
        # 显示生成结果
        for i, text in enumerate(generated_texts, 1):
            print(f"\n续写版本 {i}:")
            print(f"{selected_prompt}{text}")
            print("-" * 60)
        
        # 使用不同参数生成更有创意的版本
        print("\n生成更有创意的版本（高温度）:")
        print("-" * 60)
        
        creative_texts = generator.generate_text(
            prompt=selected_prompt,
            max_length=120,
            temperature=1.0,
            top_p=0.8,
            repetition_penalty=1.3,
            num_return_sequences=2
        )
        
        for i, text in enumerate(creative_texts, 1):
            print(f"\n创意版本 {i}:")
            print(f"{selected_prompt}{text}")
            print("-" * 60)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请确保已安装所需的依赖包：")
        print("pip install torch transformers tokenizers")

if __name__ == "__main__":
    main()
