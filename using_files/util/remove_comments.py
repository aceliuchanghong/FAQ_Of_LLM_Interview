import re
import argparse


def remove_comments(input_file, printIt=False):
    """
    移除 Python 文件中的注释（包括单行注释和多行注释）。
    :param input_file: 输入文件路径
    :param printIt: 是否打印处理后的内容
    :return: 移除注释后的文本
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ans = ""

        in_comment_block = False
        for line in lines:
            # 判断是否处于多行注释块中
            if in_comment_block:
                # 如果当前行结束了多行注释块
                if '"""' in line:
                    in_comment_block = False
                    # 去除多余的部分
                    line = line.split('"""', 1)[1]
                else:
                    # 如果还在多行注释块中，跳过当前行
                    continue

            # 处理多行注释的开始
            if '"""' in line:
                # 检查是否有未闭合的多行注释
                parts = line.split('"""')
                if len(parts) % 2 == 0:  # 偶数个 """ 表示未闭合
                    in_comment_block = True
                    line = parts[0]  # 保留注释前的内容
                else:
                    # 完整的多行注释，直接移除
                    line = parts[0] + parts[-1]

            # 处理单行注释
            line = re.sub(r"#.*", "", line)

            # 删除行尾的空白字符
            line = line.rstrip()

            # 如果行还有内容，添加到结果中
            if line:
                if printIt:
                    print(line)
                ans += line + "\n"

    return ans


if __name__ == "__main__":
    """
    python using_files/util/remove_comments.py \
        --input using_files/Reward_Guided_Code_Optimizer/train.py \
        --output 000.py
    """
    parser = argparse.ArgumentParser(description="Remove comments from a Python file.")
    parser.add_argument(
        "--input", dest="input_file", required=True, help="Input file path (required)"
    )
    parser.add_argument(
        "--output", dest="output_file", default=None, help="Output file path (optional)"
    )
    parser.add_argument(
        "--print", dest="print", action="store_true", help="Print the result to console"
    )
    opt = parser.parse_args()

    # 移除注释
    text = remove_comments(opt.input_file, opt.print)

    # 如果指定了输出文件，则保存结果
    if opt.output_file:
        with open(opt.output_file, "w", encoding="utf-8") as f:
            f.write(text)
