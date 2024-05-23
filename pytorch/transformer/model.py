import torch.nn.functional as F

from pytorch.transformer.utils import *

# 获取参数
args = parsers()
# 读取文件
text = read_book()
# 变成token
token, max_token_value = get_token(text, args.encoding)
# 获取训练的数据 ==>split_train_and_valid此处是否有失偏颇?
train_data, valid_data = split_train_and_valid(token)
# 获取训练批次张量数据
x_batch, y_batch = prepare_training_batch(train_data)
# embedding
x_batch_embedding, y_batch_embedding = getInputEmbedding(max_token_value, x_batch, y_batch)


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(args.d_model, args.d_model)
        self.Wk = nn.Linear(args.d_model, args.d_model)
        self.Wv = nn.Linear(args.d_model, args.d_model)
        # 来一个mask
        # self.register_buffer 用于向 PyTorch 模型注册一个持久化的参数，
        # 该参数在模型的训练过程中不会被更新，但可以被模型使用。在这里，它用于创建一个名为 mask 的矩阵，并将其注册为模型的缓冲区。
        self.register_buffer('mask', torch.tril(torch.ones(args.context_length, args.context_length)))

    def forward(self, x):
        # Q = x @ self.Wq
        # K = x @ self.Wk
        # V = x @ self.Wv
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # //是整数除法运算符，它会返回不大于结果的最大整数 这儿除num_heads是因为多头其实就是维度切分
        attention = Q @ K.transpose(-2, -1) / math.sqrt(args.d_model // args.num_heads)

        # Create a mask with the same shape as attention
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
        attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(args.num_heads)])
        self.projection_layer = nn.Linear(args.d_model * args.num_heads, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_rate = args.dropout

    def forward(self, x):
        attention_heads = [head(x) for head in self.heads]
        out = torch.cat(attention_heads, dim=-1)
        # self.projection_layer(out) 实际上等价于 self.projection_layer.forward(out)
        out = self.projection_layer(out)
        # self.dropout(out) 实际上等价于 self.dropout.forward(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(args.d_model)
        self.layer_norm2 = nn.LayerNorm(args.d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward_network(self.layer_norm2(x))

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_length = args.context_length
        self.token_embedding_lookup_table = nn.Embedding(max_token_value + 1, args.d_model)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock() for _ in range(args.num_blocks)] +
                [nn.LayerNorm(args.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(args.d_model, max_token_value + 1)

    def forward(self, idx, targets=None):
        # batch,token_length
        B, T = idx.shape

        position_encoding_lookup_table = torch.zeros(args.context_length, args.d_model)
        position = torch.arange(0, args.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * (-math.log(10000.0) / args.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(args.device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(split: str):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0, high=len(data) - args.context_length, size=(args.batch_size,))
    x = torch.stack([data[idx:idx + args.context_length] for idx in idxs]).to(args.device)
    y = torch.stack([data[idx + 1:idx + args.context_length + 1] for idx in idxs]).to(args.device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':
    model = Model()
    model = model.to(args.device)
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    tracked_losses = list()
    best_loss = float('inf')
    best_model_state = None
    start = time.time()
    for step in range(args.max_iters):
        losses = estimate_loss()
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            tracked_losses.append(losses)
            print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                  round(losses['valid'].item(), 3))
        if step % 2000 == 0 and step > 0:
            torch.save(model.state_dict(), f'model/model-ckpt-step{step}.pt')
        if losses['valid'].item() < best_loss:
            best_loss = losses['valid'].item()
            best_model_state = model.state_dict()
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(best_model_state, 'model/best_model-ckpt.pt')
    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f}分")
    model.eval()
    start = '宝玉和林妹妹正吃着酒,'
    encoding = tiktoken.get_encoding(args.encoding)
    start_ids = encoding.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...])
    y = model.generate(x, max_new_tokens=200)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')
