from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # 根据路由策略选择不同的网络结构
        if hasattr(args, 'moe_routing_strategy') and args.moe_routing_strategy == 'ktop1':
            # K-top-1: 使用 k 个独立的 gate networks
            self.layer = torch.nn.ModuleList([
                torch.nn.Linear(
                    args.hidden_size,
                    args.moe_num_experts,
                    bias=False,
                    dtype=common.dtype(args),
                    device=args.device
                ) for _ in range(args.moe_top_k)
            ])
            # 初始化每个 gate network
            for gate in self.layer:
                args.init_method(gate.weight)
        else:
            # 原有的单个 gate network
            self.layer = torch.nn.Linear(
                args.hidden_size,
                args.moe_num_experts,
                bias=False,
                dtype=common.dtype(args),
                device=args.device)
            args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        # K-top-1 路由策略
        if hasattr(self.args, 'moe_routing_strategy') and self.args.moe_routing_strategy == 'ktop1':
            assert isinstance(self.layer, torch.nn.ModuleList)
            assert len(self.layer) == self.args.moe_top_k
            
            # 使用 k 个独立的 gate networks
            gate_logits = torch.stack(
                [gate(x.view(-1, x.shape[-1])) for gate in self.layer],
                dim=1
            )  # [batch_size*seq_len, topk, num_experts]
            
            # 每个 gate 选择概率最高的 expert
            gate_scores = gate_logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(
                gate_scores, k=1, dim=-1
            )  # [batch_size*seq_len, topk, 1]
            
            expert_weights = expert_weights.squeeze(-1)  # [batch_size*seq_len, topk]
            expert_indices = expert_indices.squeeze(-1)  # [batch_size*seq_len, topk]
            
            # 平均权重分配
            expert_weights = expert_weights / self.args.moe_top_k
            
            # 为了保持与原有接口的兼容性，我们需要处理 scores 和 logits
            # 这里使用第一个 gate 的输出作为 scores 和 logits（可以根据需要调整）
            scores = gate_scores[:, 0, :]  # [batch_size*seq_len, num_experts]
            logits = gate_logits[:, 0, :]  # [batch_size*seq_len, num_experts]
            
        elif self.args.moe_expert_choice:
            # 原有的 expert choice 逻辑保持不变
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k
            logits = self.layer(x)
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores.transpose(1,2), (capacity * sq) // self.args.moe_num_experts, dim=-1)
        elif self.args.moe_expert_choice_grouped:
            # 原有的 grouped expert choice 逻辑保持不变
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k
            logits = self.layer(x.view(-1, x.shape[-1]))
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores.transpose(0,1),  (capacity * bs * sq) // self.args.moe_num_experts, dim=-1)
        else:
            # 原有的标准 top-k 逻辑
            logits = self.layer(x.view(-1, x.shape[-1]))
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights, p=self.args.moe_normalize_expert_weights, dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        
        return scores, logits, expert_weights, expert_indices
