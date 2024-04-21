import argparse
import torch


def parsers():
    """
    超参数设定
    :return:
    """
    parser = argparse.ArgumentParser(description="sisconsavior's model args")
    # input 初始化为 4*32*128 的矩阵
    parser.add_argument("--context_length", type=int, default=128, help="最长的token的个数")
    parser.add_argument("--d_model", type=int, default=512, help="文字的维度")
    parser.add_argument("--batch_size", type=int, default=16, help="训练批次")

    parser.add_argument("--num_blocks", type=int, default=8, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in Multi-head attention")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_iters", type=int, default=20000, help="Total of training iterations")
    parser.add_argument("--eval_interval", type=int, default=50, help="How often to evaluate the model")
    parser.add_argument("--eval_iters", type=int, default=20,
                        help="How many iterations to average the loss over when evaluating the model")
    # 设备选择参数
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="设备信息")

    args = parser.parse_args()
    return args
