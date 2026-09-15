import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import platform


# 根据操作系统自动选择中文字体
def get_chinese_font():
    system = platform.system()

    if system == "Windows":
        # Windows系统常见中文字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/STSONG.TTF',  # 华文宋体
            'C:/Windows/Fonts/STKAITI.TTF',  # 华文楷体
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # 苹方
            '/System/Library/Fonts/STHeiti Light.ttc',  # 黑体
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]

    # 查找第一个存在的字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"找到中文字体: {font_path}")
            return font_path

    print("警告：未找到系统自带中文字体，将使用默认字体（中文可能无法显示）")
    return None


def generate_wordcloud(excel_path, png_path, output_path='wordcloud.png'):
    """
    根据Excel词频数据和PNG形状生成词云图
    """

    # 1. 读取Excel文件
    print("正在读取Excel文件...")
    df = pd.read_excel(excel_path)

    # 检查列名并处理
    word_col = df.columns[0]
    freq_col = df.columns[1]

    print(f"找到词汇列: {word_col}")
    print(f"找到频次列: {freq_col}")
    print(f"共加载 {len(df)} 个词汇")

    # 2. 将数据转换为字典格式
    word_freq = dict(zip(df[word_col], df[freq_col]))
    word_freq = {word: freq for word, freq in word_freq.items() if freq > 0}
    print(f"过滤后剩余 {len(word_freq)} 个有效词汇")

    # 3. 加载形状图片
    print("正在加载形状图片...")
    mask = np.array(Image.open(png_path))

    # 4. 获取中文字体
    font_path = get_chinese_font()

    # 5. 创建词云对象（添加字体路径）
    wordcloud_config = {
        'width': 800,
        'height': 600,
        'background_color': 'white',
        'mask': mask,
        'contour_width': 0,
        'contour_color': 'black',
        'max_words': 200,
        'max_font_size': 150,
        'min_font_size': 10,
        'random_state': 42,
        'collocations': False,
        'prefer_horizontal': 0.9
    }

    # 如果找到了中文字体，添加到配置中
    if font_path:
        wordcloud_config['font_path'] = font_path

    wordcloud = WordCloud(**wordcloud_config)

    # 6. 生成词云
    print("正在生成词云...")
    wordcloud.generate_from_frequencies(word_freq)

    # 7. 保存结果
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"词云图已保存到: {output_path}")

    plt.show()
    return wordcloud


def generate_wordcloud_advanced(excel_path, png_path, output_path='wordcloud.png',
                                colors=None, bg_color='white'):
    """
    进阶版词云生成，支持自定义颜色和中文
    """

    # 读取数据
    df = pd.read_excel(excel_path)
    word_freq = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    word_freq = {word: freq for word, freq in word_freq.items() if freq > 0}

    # 加载形状
    mask = np.array(Image.open(png_path))

    # 获取中文字体
    font_path = get_chinese_font()

    # 基础配置
    wc_config = {
        'width': 800,
        'height': 600,
        'background_color': bg_color,
        'mask': mask,
        'contour_width': 0,
        'max_words': 200,
        'max_font_size': 150,
        'min_font_size': 10,
        'random_state': 42,
        'collocations': True,
        'prefer_horizontal': 0.62
    }

    # 添加中文字体
    if font_path:
        wc_config['font_path'] = font_path

    # 如果指定了颜色，添加颜色函数
    if colors:
        from random import choice
        def custom_color_func(word, font_size, position, orientation,
                              random_state=None, **kwargs):
            return choice(colors)

        wc_config['color_func'] = custom_color_func

    # 生成词云
    wordcloud = WordCloud(**wc_config)
    wordcloud.generate_from_frequencies(word_freq)

    # 保存
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    return wordcloud


def generate_wordcloud_demo():
    """
    演示函数：使用示例数据生成词云
    """

    excel_file = "myexcel.xlsx"
    png_file = "cbg.png"
    output_file = "词云图结果.png"

    if not os.path.exists(excel_file):
        print(f"错误：找不到Excel文件 '{excel_file}'")
        return

    if not os.path.exists(png_file):
        print(f"错误：找不到PNG图片 '{png_file}'")
        return

    try:
        # 你可以选择使用基础版或进阶版
        # 基础版
        generate_wordcloud(excel_file, png_file, output_file)

        # 或者使用进阶版（自定义颜色）
        # custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194']
        # generate_wordcloud_advanced(excel_file, png_file, '彩色词云.png',
        #                            colors=custom_colors, bg_color='white')

        print("词云生成成功！")
    except Exception as e:
        print(f"生成词云时出错：{e}")


if __name__ == "__main__":
    generate_wordcloud_demo()