import argparse
import torch


def parsers():
    parser = argparse.ArgumentParser(description="sisconsavior's model args")
    parser.add_argument("--context_length", type=int, default=256,
                        help="上下文长度是指在处理文本数据时，模型可以看到的最大序列长度")
    parser.add_argument("--d_model", type=int, default=512, help="文字的维度")
    parser.add_argument("--batch_size", type=int, default=128, help="每一次训练的数据大小")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in Multi-head attention")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_iters", type=int, default=8001, help="Total of training iterations")
    parser.add_argument("--eval_interval", type=int, default=50, help="How often to evaluate the model")
    parser.add_argument("--eval_iters", type=int, default=20,
                        help="How many iterations to average the loss over when evaluating the model")
    parser.add_argument("--num_epochs", type=int, default=10, help="迭代次数")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="设备信息")
    parser.add_argument("--data_path", type=str, default='../data/books/', help="Data path")
    parser.add_argument("--encoding", type=str, default='cl100k_base', help="Encoding scheme")
    args = parser.parse_args()
    return args
