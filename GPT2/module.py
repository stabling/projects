from header import *
import config as cfg


class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask

        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)

        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        if self.isMask:
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, x):
        x = self.c_attn(x)
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)
        x = x.transpose(-2, -3)
        q, k, v = x.chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2)) / self.dk
        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]
        w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)

        a = w @ v

        a = a.transpose(-2, -3)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)

        h = self.c_proj(a)
        h = self.resi_drop(h)

        return h


class Block(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()

        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)

        self.attention = Attention(isMask)

        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        a = a + x
        a = self.layer_normal_2(a)
        h = self.proj(a)
        h = self.dropout(h)
        y = h + a
        return y


class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)
        self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)

        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block())

        self.drop = nn.Dropout(0.1)

        self.sequential = nn.Sequential(*self.blocks)

        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, x, p, t):
        e = self.vocab_embed(x)
        p = self.pos_embed(p)
        t = self.type_embed(t)
        h = self.drop(e + p + t)
        h = self.sequential(h)
        return self.output_layer(h)


if __name__ == '__main__':
    # x = torch.randint(0, 4000, size=(2, cfg.pos_num)).to(torch.device(cfg.device))
    # p = torch.arange(0, cfg.pos_num)[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
    #
    gpt = GPT2()
    gpt.to(torch.device(cfg.device))
    gpt.eval()
    # gpt.load_state_dict(torch.load("weights/apt2_k.pt"))
    # y = gpt(x, p)
    # print(y.shape)
    #
    # print(sum(param.numel() for param in block.parameters()))

    os = []
    x = torch.tensor([[0]]).cuda()
    p = torch.tensor([[0]]).cuda()
    for i in range(100):
        y = gpt(x, p)
        y = y[:, -1:]
        v, y = torch.topk(y, 8, dim=-1)

        v, y = v.reshape(-1, 8), y.reshape(-1, 8)
        v = torch.multinomial(torch.softmax(v, dim=-1), 1)
        y = torch.gather(y, -1, v)

        x = torch.cat([x, y], dim=1)
        p = torch.tensor([range(i + 2)]).cuda()

        print(x)
