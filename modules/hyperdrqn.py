import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == in_dim, "in_dim must be divisible by num_heads"

        # 线性变换，将输入投影到多个头的特征
        self.linear = nn.Linear(in_dim, num_heads * self.head_dim)

        # 注意力参数a，每个头对应一个向量，分解为左右两部分
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim))
        nn.init.xavier_normal_(self.a)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.out_proj = nn.Linear(num_heads * self.head_dim, in_dim)

    def forward(self, x, adj_mask):
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # 线性变换并分头
        Wh = self.linear(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]

        # 拆分注意力参数a为左右两部分
        a_left = self.a[:, :d].unsqueeze(-1)  # [H, d, 1]
        a_right = self.a[:, d:].unsqueeze(-1)  # [H, d, 1]

        # 计算每个节点的左、右注意力得分
        h_left = torch.matmul(Wh, a_left).squeeze(-1)  # [B, H, N]
        h_right = torch.matmul(Wh, a_right).squeeze(-1)  # [B, H, N]

        # 生成所有节点对的注意力分数
        e = h_left.unsqueeze(-1) + h_right.unsqueeze(2)  # [B, H, N, N]
        e = self.leaky_relu(e)

        # 应用邻接矩阵mask
        adj_mask = adj_mask.unsqueeze(1)  # [B, 1, N, N]
        e = e.masked_fill(adj_mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(e, dim=-1)

        # 加权求和
        out = torch.matmul(attn_weights, Wh)  # [B, H, N, d]

        # 合并多头并投影回原维度
        out = out.transpose(1, 2).contiguous().view(B, N, H * d)
        out = self.out_proj(out)

        return out, attn_weights


class HyperDRQN(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, num_heads=4, skip_connect=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.skip_connect = skip_connect

        # Spatial encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Graph attention
        self.graph_attn = GraphAttention(hidden_dim, num_heads)

        # Temporal module
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Skip connection
        if skip_connect:
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x, adj_mask, hidden_state):
        B, N, D = x.shape

        # Spatial encoding
        h_spatial = self.obs_encoder(x)  # [B, N, H]

        # Add positional embedding
        h_spatial += self.pos_embed.expand(B, N, -1)

        # Graph attention with adjacency mask
        h_attn, attn_weights = self.graph_attn(h_spatial, adj_mask)

        # Skip connection
        if self.skip_connect:
            h_attn = h_attn + self.skip_proj(h_spatial)

        # Temporal processing
        h_attn = h_attn.view(B * N, -1)
        hidden_state = hidden_state.view(B * N, -1)
        h_temp = self.gru(h_attn, hidden_state).view(B, N, -1)

        # Decoding
        q_values = self.decoder(h_temp)
        return q_values, h_temp

    # from torchviz import make_dot
    # # 假设已定义模型和输入示例
    # x = torch.randn(2, 10, 64)  # 示例输入
    # adj_mask = torch.ones(2, 10, 10)
    # model = HyperDRQN(in_dim=64, hidden_dim=32, action_dim=4)
    # q_values, _ = model(x, adj_mask, hidden_state=torch.zeros(2, 10, 32))
    # dot = make_dot(q_values, params=dict(model.named_parameters()))
    # dot.render("HyperDRQN_structure")  # 生成PDF或PNG