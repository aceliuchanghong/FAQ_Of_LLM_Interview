import torch
import tiktoken

from model import Model
from pytorch.transformer.config import parsers

arg = parsers()
model_path = 'model/best_model-ckpt.pt'
device = arg.device

# Load the pre-trained model
model = Model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


def generate_text(prompt, max_new_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    start_ids = encoding.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, max_new_tokens)
    generated_text = encoding.decode(y[0].tolist())
    return generated_text


prompt = '黛玉道：“你来做什么？”宝玉笑道：'
prompt2 = '第五十回 薛小妹新编怀古诗'
max_new_tokens = 300
generated_text = generate_text(prompt2, max_new_tokens)
print('---------------')
print(generated_text)
print('---------------')
