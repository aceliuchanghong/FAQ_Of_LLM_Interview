### What it is
Transformer 是一种神经网络结构，它利用了自注意力（self-attention）机制和多层编码器（encoder）与解码器（decoder）层
从而有效地处理长距离依赖关系和捕获不同层次的文本信息。

---

#### 以下是现代Transformer大模型的常用模块总结：

1. **Tokenizer（分词器）**：将文本切分成token，映射为ID。
2. **Embedding（嵌入层）**：将token ID转为d维向量，捕捉语义。
3. **Positional Encoding（位置编码）**：为每个token添加位置信息。
4. **Encoder（编码器，N层堆叠）**：
   - **Multi-Head Attention（多头注意力，MHA）**：捕捉token间全局依赖。
   - **LayerNorm（层归一化）**：稳定MHA输入/输出。
   - **Feed-Forward Network（FFN，前馈网络）**：对每个token做非线性变换。
   - **LayerNorm（层归一化）**：稳定FFN输出。
   - **Residual Connection（残差连接）**：MHA和FFN后加回输入。
5. **Decoder（解码器，N层堆叠，若有）**：
   - **Masked Multi-Head Attention（掩码多头注意力）**：只关注当前及之前token。
   - **LayerNorm**：稳定掩码MHA。
   - **Cross-Attention（交叉注意力）**：关注编码器输出。
   - **LayerNorm**：稳定交叉注意力。
   - **Feed-Forward Network（FFN）**：同编码器FFN。
   - **LayerNorm**：稳定FFN。
   - **Residual Connection**：同编码器。
6. **Linear + Softmax（输出层）**：将解码器输出映射为词汇表概率，预测token。

---

#### 模块代码更新见下方

![img.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fwrite%2Fimg.png)

![img_11.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fimg_11.png)

![img_1.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fimg_1.png)

![img_1.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fwrite%2Fimg_1.png)

![img_12.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fimg_12.png)

model.py

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, d_model, head_size, context_length, dropout=0.1, bias=False):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.bias = bias
        self.Wq = nn.Linear(d_model, head_size, bias=self.bias)
        self.Wk = nn.Linear(d_model, head_size, bias=self.bias)
        self.Wv = nn.Linear(d_model, head_size, bias=self.bias)

        self.register_buffer(
            "mask", torch.tril(torch.ones(self.context_length, self.context_length))
        )
        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # [B, (TimeStep) context_length, head_size] ==> 由于是一个个token预测,所以输入的是随步骤增加而增加的
        B, T, D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        output = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        output = output.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        output = F.softmax(output, dim=-1)
        output = self.Dropout(output)

        output = output @ v

        return output


class MHA(nn.Module):
    def __init__(self, d_model, head_size, context_length, nums_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.nums_head = nums_head
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [
                Attention(
                    self.d_model, self.head_size, self.context_length, self.dropout
                )
                for _ in range(self.nums_head)
            ]
        )
        self.Wo = nn.Linear(self.d_model, self.d_model)
        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.Wo(output)
        output = self.Dropout(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_size, context_length, nums_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.nums_head = nums_head
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.mha = MHA(
            self.d_model,
            self.head_size,
            self.context_length,
            self.nums_head,
            self.dropout,
        )
        self.ffn = FeedForwardNetwork(self.d_model, self.dropout)

    def forward(self, x):
        output = self.ln1(x)
        output = self.mha(output)
        output = x + output

        output = self.ln2(output)
        output = self.ffn(output)
        output = x + output

        return output


class Model(nn.Module):
    def __init__(
        self,
        num_blocks,
        max_token_value,
        d_model,
        head_size,
        context_length,
        nums_head,
        device,
        dropout=0.1,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.max_token_value = max_token_value
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.nums_head = nums_head
        self.device = device
        self.dropout = dropout

        self.vocab_linear = nn.Linear(self.d_model, self.max_token_value + 1)
        self.token_emb_lookup_table = nn.Embedding(
            self.max_token_value + 1, self.d_model
        )
        self.transformer_block = nn.Sequential(
            *(
                [
                    TransformerBlock(
                        self.d_model,
                        self.head_size,
                        self.context_length,
                        self.nums_head,
                        self.dropout,
                    )
                    for _ in range(self.num_blocks)
                ]
                + [nn.LayerNorm(self.d_model)]
            )
        )

        # 预计算位置编码
        position_encoding_lookup_table = torch.zeros(
            self.context_length, self.d_model, device=self.device
        )
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "position_encoding_lookup_table", position_encoding_lookup_table
        )

    def forward(self, x_batch, y_batch=None):
        # x_batch ==> [B, (TimeStep) context_length]
        B, T = x_batch.shape
        # 动态选择与输入序列长度 T 匹配的位置编码
        pos_enc = self.position_encoding_lookup_table[:T, :]

        output = self.token_emb_lookup_table(x_batch) + pos_enc
        output = self.transformer_block(output)
        logits = self.vocab_linear(output)

        if y_batch is not None:
            B, T, D = logits.shape
            logits_reshape = logits.view(B * T, D)
            y_batch_reshape = y_batch.view(B * T)
            loss = F.cross_entropy(input=logits_reshape, target=y_batch_reshape)
        else:
            loss = None

        return logits, loss

    def generate(self, x_batch, temprature=1.0, max_tokens=100, num_samples=1):
        # x_batch.shape ==> [B, (TimeStep) context_length]
        for _ in range(max_tokens):
            x_crop = x_batch[:, -1 * self.context_length :]
            # logits.shape ==> [(1) B, (TimeStep) context_length, max_token_value]
            logits, loss = self.forward(x_crop)
            logits = logits[:, -1, :] / temprature
            probabilities = F.softmax(logits, dim=-1)
            predict_token = torch.multinomial(probabilities, num_samples=num_samples)
            x_batch = torch.cat((x_batch, predict_token), dim=1)

        return x_batch
```

train.py
```python
from model import Model

def get_batch(
    tokenized_text, split="train", context_length=16, batch_size=32, part=0.9
):
    split_size = int(len(tokenized_text) * part)
    train_data = tokenized_text[:split_size]
    valid_data = tokenized_text[split_size:]
    if split == "train":
        data = train_data
    else:
        data = valid_data

    idxs = torch.randint(0, len(data) - context_length - 1, size=(batch_size,))
    x_batch = torch.stack([data[idx : idx + context_length] for idx in idxs])
    y_batch = torch.stack([data[idx + 1 : idx + 1 + context_length] for idx in idxs])

    return x_batch, y_batch

@torch.no_grad()
def estimate_loss(model, eval_iters, tokenized_text, context_length, batch_size):
    model.eval()
    out = {}
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(
                tokenized_text,
                split=split,
                context_length=context_length,
                batch_size=batch_size,
            )
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    num_blocks = 12
    d_model = 1024
    context_length = 64
    nums_head = 8
    head_size = d_model // nums_head
    dropout = 0.1
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    max_iters = 10000
    learning_rate = 1e-3
    eval_interval = 50
    eval_iters = 10
    TORCH_SEED = 1337
    torch.manual_seed(TORCH_SEED)

    with open("../../z_using_files/txt/sales_textbook.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenized_text = tokenizer.encode(text)
    tokenized_text = torch.tensor(data=tokenized_text, dtype=torch.long, device=device)
    max_token_value = tokenized_text.max().item()

    model = Model(
        num_blocks,
        max_token_value,
        d_model,
        head_size,
        context_length,
        nums_head,
        device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for step in range(max_iters):

        if (step + 1) % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model, eval_iters, tokenized_text, context_length, batch_size
            )
            logger.info(
                colored(
                    f"Step:{step+1}, Train loss:{round(losses['train'].item(),3)}, Valid loss:{round(losses['valid'].item(),3)}",
                    "green",
                )
            )

        x_train_batch_data, y_train_batch_data = get_batch(
            tokenized_text,
            context_length=context_length,
            batch_size=batch_size,
        )
        logits, loss = model(x_train_batch_data, y_train_batch_data)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            torch.save(model.state_dict(), f"model_{step+1}.ckpt")

    torch.save(model.state_dict(), "model.ckpt")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(colored(f"耗时: {elapsed_time:.2f}秒", "green"))
```

inference.py
```python
from model import Model

if __name__ == "__main__":
    """
    效果如下:
    输入语句:
    By focusing on providing clear and concise explanations, sales professionals can effectively:

    输出:
    convey ideas and information without overwhelming or alienating potential customers.
    2. Articulation and Pronunciation:
    Clear articulation and proper pronunciation are vital in verbal communication. Speaking clearly with well-enunciated words not only enhances the listeners' understanding but also establishes credibility and professionalism. By practicing and improving our articulation and pronunciation, we can ensure that our message is delivered with confidence and clarity.
    3. Voice Modulation and Tone:
    While ignoring body language and tone of voice in this discussion, it is
    """
    num_blocks = 12
    d_model = 1024
    context_length = 64
    nums_head = 8
    head_size = d_model // nums_head
    dropout = 0.1
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    max_token_value = 100069

    checkpoint = torch.load("model.ckpt")
    model = Model(
        num_blocks,
        max_token_value,
        d_model,
        head_size,
        context_length,
        nums_head,
        device,
    )
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    text = "hello, who are you?"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenized_text = tokenizer.encode(text)

    # The [None, ...] adds a batch dimension to make the tensor compatible with the model's expected input format.
    x = torch.tensor(tokenized_text, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x)
        logger.info(colored(f"{tokenizer.decode(y[0].tolist())}", "green"))
```

##### 模型训练结果
```
uv run train.py

2025-04-16 22:01:52-INFO: Step:0, Train loss:11.683, Valid loss:11.679
    ...
2025-04-16 22:05:48-INFO: Step:450, Train loss:0.514, Valid loss:5.878
    ...
2025-04-16 23:08:38-INFO: Step:6100, Train loss:0.133, Valid loss:9.498
    ...
模型大小:1.33G
```

### Reference(参考文档)
* [论文:Attention Is All You Need](..%2Fusing_files%2Fpaper%2Fpytorch2transformer.pdf)
* [原始论文地址](https://arxiv.org/abs/1706.03762)
* [llm张-教学视频](https://space.bilibili.com/3546611527453161/lists/4721368)
* 