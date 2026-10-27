"""
patches.py  (inference/common)
--------------------------------
All model-patching helpers shared across the deploy and faithful tracks.

Placing patch logic here (rather than in deploy/export_onnx.py) keeps it
accessible to the faithful track without creating a cross-track dependency.

Public API
----------
patch_natten_for_onnx          -- NATTEN → global nn.MultiheadAttention (ONNX-safe)
patch_natten_faithful          -- NATTEN → local-window MHA (PyTorch-only, approximate)
patch_natten_exact             -- NATTEN → exact gather-based re-implementation
                                  (ONNX-safe, keeps RPB and NATTEN's edge windows)
patch_mha_for_onnx             -- batch_first=True MHA → batch_first=False wrapper
patch_boolean_indexing_for_onnx -- bool-indexing → mask-multiply / torch.where
patch_agent_encoder_for_onnx   -- AgentEncoder bool-gather → gate-multiply
patch_planning_model_for_onnx  -- atan2 → ONNX-exportable approximation
apply_onnx_export_patches      -- every patch above, as applied before export

Internal wrapper classes (also importable if needed):
_MHAWrapper        -- drop-in for NeighborhoodAttention1D (global MHA)
_LocalWindowMHA    -- drop-in for NeighborhoodAttention1D (local-window MHA)
_ExactNA1D         -- drop-in for NeighborhoodAttention1D (exact, ONNX-safe)
_Bf1MHAWrapper     -- ONNX-safe re-implementation of existing MHA modules
                      (key_padding_mask kept as an additive float mask)
"""

import os
import sys

_here       = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root  = os.path.dirname(_inference)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# _MHAWrapper — global MHA drop-in for NeighborhoodAttention1D (ONNX-safe)
# ---------------------------------------------------------------------------

class _MHAWrapper(nn.Module):
    """Drop-in for NeighborhoodAttention1D: (B, L, C) → (B, L, C).

    Uses batch_first=False to avoid PyTorch 1.12's fused
    _native_multi_head_attention kernel, which is not ONNX exportable.
    """

    def __init__(self, dim: int, num_heads: int, qkv_weight, qkv_bias, proj_weight, proj_bias):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=True,
            batch_first=False,  # (L, B, C) path — ONNX exportable
        )
        # NeighborhoodAttention stores qkv as one weight [3*dim, dim].
        # nn.MultiheadAttention stores in_proj_weight [3*dim, dim] (same layout).
        with torch.no_grad():
            self.mha.in_proj_weight.copy_(qkv_weight)
            self.mha.in_proj_bias.copy_(qkv_bias)
            self.mha.out_proj.weight.copy_(proj_weight)
            self.mha.out_proj.bias.copy_(proj_bias)

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(0, 1)                        # (L, B, C)
        out, _ = self.mha(x, x, x, need_weights=False)
        return out.transpose(0, 1)                   # (B, L, C)


# ---------------------------------------------------------------------------
# _LocalWindowMHA — faithful local-window drop-in (PyTorch-only)
# ---------------------------------------------------------------------------

class _LocalWindowMHA(nn.Module):
    """
    Faithful approximation of NeighborhoodAttention1D.

    Each token attends ONLY to its kernel_size nearest neighbours (mirroring
    NATTEN's neighbourhood pattern) instead of all tokens.  Weights (qkv, proj)
    are copied exactly from the original NATTEN module.  The relative positional
    bias (RPB) is still dropped — it requires a custom ONNX op.

    Attention complexity: O(L * kernel_size) vs O(L^2) for global MHA.

    Level 2 of NATSequenceEncoder has sequence length 5 and kernel_size 5,
    so that level is already globally equivalent even with this wrapper.

    This is NOT ONNX-exportable with PyTorch 1.12 due to `unfold`, but it
    provides much higher PyTorch fidelity than plain global MHA for levels 0/1.

    Two differences from NATTEN remain: the RPB is dropped, and edge tokens
    attend to zero-padded keys/values, whereas NATTEN shifts the window inward
    so every token sees kernel_size real neighbours. _ExactNA1D fixes both.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.half_k = kernel_size // 2

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        with torch.no_grad():
            self.qkv.weight.copy_(qkv_weight)
            self.qkv.bias.copy_(qkv_bias)
            self.proj.weight.copy_(proj_weight)
            self.proj.bias.copy_(proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim
        pad = self.half_k

        # Q, K, V projections
        qkv = self.qkv(x)                                    # [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)                       # each [B, L, C]

        # Reshape to [B, H, L, D]
        q = q.reshape(B, L, H, D).permute(0, 2, 1, 3)       # [B, H, L, D]
        k = k.reshape(B, L, H, D).permute(0, 2, 1, 3)
        v = v.reshape(B, L, H, D).permute(0, 2, 1, 3)

        # Pad K, V along sequence dim so every position has kernel_size neighbours
        k_pad = F.pad(k, (0, 0, pad, pad))                   # [B, H, L+2p, D]
        v_pad = F.pad(v, (0, 0, pad, pad))

        # unfold(dim, size, step) → slides window of `size` along `dim`
        k_win = k_pad.unfold(2, self.kernel_size, 1)         # [B, H, L, D, ks]
        v_win = v_pad.unfold(2, self.kernel_size, 1)

        k_win = k_win.transpose(-1, -2)                      # [B, H, L, ks, D]
        v_win = v_win.transpose(-1, -2)

        attn = torch.matmul(
            q.unsqueeze(3),                                  # [B, H, L, 1, D]
            k_win.transpose(-1, -2),                         # [B, H, L, D, ks]
        ) * self.scale                                        # [B, H, L, 1, ks]
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v_win).squeeze(3)           # [B, H, L, D]
        out = out.permute(0, 2, 1, 3).reshape(B, L, C)      # [B, L, C]
        return self.proj(out)


# ---------------------------------------------------------------------------
# _ExactNA1D — exact NATTEN-1D re-implementation (ONNX-safe)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _na1d_indices(length: int, kernel_size: int):
    """Neighbour and relative-position-bias indices used by NATTEN 1D.

    Mirrors NATTEN's get_window_start / get_pb_start for dilation 1: a token
    near either edge keeps a full window of kernel_size real neighbours by
    shifting the window inward, and the RPB index shifts with it.

    Returns two LongTensors of shape [length, kernel_size].
    """
    nb = kernel_size // 2
    win, pb = [], []
    for i in range(length):
        start = max(i - nb, 0) + (i + nb >= length) * (length - i - nb - 1)
        pb_start = nb + (i < nb) * (nb - i) + (i + nb >= length) * (length - i - 1 - nb)
        win.append([start + j for j in range(kernel_size)])
        pb.append([pb_start + j for j in range(kernel_size)])
    return torch.tensor(win, dtype=torch.long), torch.tensor(pb, dtype=torch.long)


class _ExactNA1D(nn.Module):
    """
    Drop-in for NeighborhoodAttention1D (dilation 1) that reproduces its output
    to float precision, using only gather / multiply / softmax.

    Unlike _MHAWrapper and _LocalWindowMHA it keeps the relative positional
    bias and NATTEN's edge handling, so no accuracy is traded for
    exportability. The neighbour indices depend only on the sequence length,
    which is fixed for a given input shape, so they are baked into the ONNX
    graph as constants.
    """

    def __init__(self, nat_attn: nn.Module):
        super().__init__()
        if getattr(nat_attn, "dilation", 1) != 1:
            raise NotImplementedError("_ExactNA1D only supports dilation=1")
        self.num_heads = nat_attn.num_heads
        self.kernel_size = nat_attn.kernel_size
        self.scale = nat_attn.scale
        # Reuse the trained parameters directly (no copy needed).
        self.qkv = nat_attn.qkv
        self.proj = nat_attn.proj
        self.rpb = nat_attn.rpb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Lp, C = x.shape
        H = self.num_heads
        D = C // H
        ks = self.kernel_size

        # NATTEN zero-pads inputs shorter than the window (before qkv).
        pad_r = max(0, ks - Lp)
        if pad_r:
            x = F.pad(x, (0, 0, 0, pad_r))
        L = x.shape[1]

        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]   # each [B, H, L, D]

        win, pb = _na1d_indices(L, ks)
        win = win.to(x.device)
        k_win = k.index_select(2, win.flatten()).view(B, H, L, ks, D)
        v_win = v.index_select(2, win.flatten()).view(B, H, L, ks, D)

        attn = (q.unsqueeze(3) * k_win).sum(-1)          # [B, H, L, ks]
        if self.rpb is not None:
            attn = attn + self.rpb[:, pb.to(x.device)]   # [H, L, ks]
        attn = attn.softmax(dim=-1)

        out = (attn.unsqueeze(-1) * v_win).sum(3)        # [B, H, L, D]
        out = out.permute(0, 2, 1, 3).reshape(B, L, C)
        if pad_r:
            out = out[:, :Lp]
        return self.proj(out)


# ---------------------------------------------------------------------------
# _Bf1MHAWrapper — batch_first=False wrapper for existing MHA modules
# ---------------------------------------------------------------------------

class _Bf1MHAWrapper(nn.Module):
    """
    ONNX-safe replacement for an existing nn.MultiheadAttention
    (batch_first=True, inference only).

    Computes attention explicitly instead of calling nn.MultiheadAttention,
    which avoids PyTorch 1.12's fused _native_multi_head_attention kernel and
    the exporter's bool-mask Expand bug. key_padding_mask is applied as an
    additive float mask, so padded agents / map polygons are still ignored.

    (An earlier version dropped key_padding_mask entirely. That is invisible
    with all-valid dummy inputs but changes outputs whenever a scene has
    fewer than max agents or polygons, i.e. in practically every real scene.)
    """

    _MASK_VALUE = -1e9

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        if mha.bias_k is not None or not mha._qkv_same_embed_dim:
            raise NotImplementedError("_Bf1MHAWrapper needs packed qkv without bias_kv")
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if attn_mask is not None:
            raise NotImplementedError("attn_mask is not used by PlanTF")
        B, Lq, C = query.shape
        Lk = key.shape[1]
        H, D = self.num_heads, self.head_dim

        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, w_q, b_q).view(B, Lq, H, D).transpose(1, 2)  # [B, H, Lq, D]
        k = F.linear(key, w_k, b_k).view(B, Lk, H, D).transpose(1, 2)    # [B, H, Lk, D]
        v = F.linear(value, w_v, b_v).view(B, Lk, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * (D ** -0.5)        # [B, H, Lq, Lk]
        if key_padding_mask is not None:
            attn = attn + key_padding_mask.view(B, 1, 1, Lk).float() * self._MASK_VALUE
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Lq, C)
        return self.out_proj(out), None


# ---------------------------------------------------------------------------
# Patch functions
# ---------------------------------------------------------------------------

def patch_natten_for_onnx(model: nn.Module) -> None:
    """
    Replace every NeighborhoodAttention1D inside NATLayer blocks with a
    standard nn.MultiheadAttention (weights copied, RPB dropped).

    Done IN-PLACE. Does NOT affect checkpoints or the nuPlan simulation path.
    """
    from src.models.planTF.layers.embedding import NATLayer

    patched = 0
    for module in model.modules():
        if isinstance(module, NATLayer):
            nat_attn = module.attn
            dim = nat_attn.qkv.weight.shape[1]
            num_heads = nat_attn.num_heads
            wrapper = _MHAWrapper(
                dim=dim,
                num_heads=num_heads,
                qkv_weight=nat_attn.qkv.weight.data,
                qkv_bias=nat_attn.qkv.bias.data,
                proj_weight=nat_attn.proj.weight.data,
                proj_bias=nat_attn.proj.bias.data,
            )
            module.attn = wrapper
            patched += 1
    print(f"patch_natten_for_onnx: replaced {patched} NATLayer attention blocks")


def patch_natten_faithful(model: nn.Module) -> None:
    """
    Replace NeighborhoodAttention1D with _LocalWindowMHA — preserves windowed
    neighbourhood pattern.  NOT ONNX-exportable (uses unfold).
    """
    from src.models.planTF.layers.embedding import NATLayer

    patched = 0
    for module in model.modules():
        if isinstance(module, NATLayer):
            nat_attn = module.attn
            dim = nat_attn.qkv.weight.shape[1]
            module.attn = _LocalWindowMHA(
                dim=dim,
                num_heads=nat_attn.num_heads,
                kernel_size=nat_attn.kernel_size,
                qkv_weight=nat_attn.qkv.weight.data,
                qkv_bias=nat_attn.qkv.bias.data,
                proj_weight=nat_attn.proj.weight.data,
                proj_bias=nat_attn.proj.bias.data,
            )
            patched += 1
    print(
        f"patch_natten_faithful: replaced {patched} NATLayer blocks with "
        f"local-window MHA (kernel_size preserved, RPB dropped)"
    )


def patch_natten_exact(model: nn.Module) -> None:
    """
    Replace NeighborhoodAttention1D with _ExactNA1D — numerically equivalent
    to NATTEN (RPB and edge windows preserved) and ONNX-exportable.
    """
    from src.models.planTF.layers.embedding import NATLayer

    patched = 0
    for module in model.modules():
        if isinstance(module, NATLayer):
            module.attn = _ExactNA1D(module.attn)
            patched += 1
    print(f"patch_natten_exact: replaced {patched} NATLayer blocks with exact NA1D")


def patch_mha_for_onnx(model: nn.Module) -> None:
    """
    Replace batch_first=True nn.MultiheadAttention modules in
    TransformerEncoderLayer and StateAttentionEncoder with _Bf1MHAWrapper
    (explicit attention; key_padding_mask kept for TransformerEncoderLayer).

    StateAttentionEncoder only builds a key_padding_mask in training mode
    (state dropout), so at inference it has none to pass.
    """
    import types
    from src.models.planTF.layers.transformer_encoder_layer import TransformerEncoderLayer
    from src.models.planTF.modules.agent_encoder import StateAttentionEncoder

    patched = 0

    for module in model.modules():
        if isinstance(module, TransformerEncoderLayer):
            module.attn = _Bf1MHAWrapper(module.attn)

            def _fwd(self, src, mask=None, key_padding_mask=None):
                src2 = self.norm1(src)
                src2, _ = self.attn(
                    src2, src2, src2,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                src = src + self.drop_path1(src2)
                src = src + self.drop_path2(self.mlp(self.norm2(src)))
                return src

            module.forward = types.MethodType(_fwd, module)
            patched += 1

        elif isinstance(module, StateAttentionEncoder):
            module.attn = _Bf1MHAWrapper(module.attn)

            def _sea_fwd(self, x):
                x_embed = []
                for i, linear in enumerate(self.linears):
                    x_embed.append(linear(x[:, i, None]))
                x_embed = torch.stack(x_embed, dim=1)
                pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
                x_embed = x_embed + pos_embed
                query = self.query.repeat(x_embed.shape[0], 1, 1)
                x_state, _ = self.attn(query, x_embed, x_embed, need_weights=False)
                return x_state[:, 0]

            module.forward = types.MethodType(_sea_fwd, module)
            patched += 1

    print(f"patch_mha_for_onnx: patched {patched} MHA modules (key_padding_mask kept)")


def patch_boolean_indexing_for_onnx(model: nn.Module) -> None:
    """
    Replace boolean-advanced-indexing operations that block ONNX tracing.

    1. PointsEncoder.forward — x[bool_mask] → mask-multiply.
    2. MapEncoder.forward    — x[bool_mask]=y → torch.where.
    """
    import types
    from src.models.planTF.layers.embedding import PointsEncoder
    from src.models.planTF.modules.map_encoder import MapEncoder

    patched = 0

    for module in model.modules():

        if isinstance(module, PointsEncoder):
            def _pe_fwd(self, x, mask=None):
                bs, n, c = x.shape
                if mask is not None:
                    mf = mask.float().unsqueeze(-1)
                    x = x * mf
                x_flat = x.reshape(bs * n, c)
                x_mlp = self.first_mlp(x_flat).view(bs, n, 256)
                if mask is not None:
                    x_mlp = x_mlp * mf
                pooled = x_mlp.max(dim=1)[0]
                x_cat = torch.cat(
                    [x_mlp, pooled.unsqueeze(1).expand(-1, n, -1)], dim=-1
                )
                x_cat_flat = x_cat.reshape(bs * n, 512)
                res = self.second_mlp(x_cat_flat).view(bs, n, self.encoder_channel)
                if mask is not None:
                    res = res * mf
                return res.max(dim=1)[0]

            module.forward = types.MethodType(_pe_fwd, module)
            patched += 1

        elif isinstance(module, MapEncoder):
            def _me_fwd(self, data):
                polygon_center          = data["map"]["polygon_center"]
                polygon_type            = data["map"]["polygon_type"].long()
                polygon_on_route        = data["map"]["polygon_on_route"].long()
                polygon_tl_status       = data["map"]["polygon_tl_status"].long()
                polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
                polygon_speed_limit     = data["map"]["polygon_speed_limit"]
                point_position          = data["map"]["point_position"]
                point_vector            = data["map"]["point_vector"]
                point_orientation       = data["map"]["point_orientation"]
                valid_mask              = data["map"]["valid_mask"]

                polygon_feature = torch.cat([
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack([
                        point_orientation[:, :, 0].cos(),
                        point_orientation[:, :, 0].sin(),
                    ], dim=-1),
                ], dim=-1)

                bs, M, P, C = polygon_feature.shape
                valid_mask_flat     = valid_mask.view(bs * M, P)
                polygon_feature_flat = polygon_feature.reshape(bs * M, P, C)
                x_polygon = self.polygon_encoder(
                    polygon_feature_flat, valid_mask_flat
                ).view(bs, M, -1)

                x_type      = self.type_emb(polygon_type)
                x_on_route  = self.on_route_emb(polygon_on_route)
                x_tl_status = self.traffic_light_emb(polygon_tl_status)

                speed_feat   = self.speed_limit_emb(polygon_speed_limit.unsqueeze(-1))
                unknown_feat = self.unknown_speed_emb.weight.view(1, 1, -1).expand(bs, M, -1)
                x_speed_limit = torch.where(
                    polygon_has_speed_limit.unsqueeze(-1), speed_feat, unknown_feat
                )

                x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit
                return x_polygon

            module.forward = types.MethodType(_me_fwd, module)
            patched += 1

    print(f"patch_boolean_indexing_for_onnx: patched {patched} modules")


def patch_agent_encoder_for_onnx(model: nn.Module) -> None:
    """
    AgentEncoder.forward uses x[bool_mask] gather/scatter for history_encoder.
    Replace with: run encoder on ALL agents, zero-gate invalid agents.
    """
    import types
    from src.models.planTF.modules.agent_encoder import AgentEncoder

    patched = 0
    for module in model.modules():
        if isinstance(module, AgentEncoder):

            def _ae_fwd(self, data):
                T          = self.hist_steps
                position   = data["agent"]["position"][:, :, :T]
                heading    = data["agent"]["heading"][:, :, :T]
                velocity   = data["agent"]["velocity"][:, :, :T]
                shape      = data["agent"]["shape"][:, :, :T]
                category   = data["agent"]["category"].long()
                valid_mask = data["agent"]["valid_mask"][:, :, :T]

                heading_vec    = self.to_vector(heading, valid_mask)
                valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
                agent_feature  = torch.cat([
                    self.to_vector(position, valid_mask),
                    self.to_vector(velocity, valid_mask),
                    torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                    shape[:, :, 1:],
                    valid_mask_vec.float().unsqueeze(-1),
                ], dim=-1)

                bs, A, Tm1, _ = agent_feature.shape
                agent_flat = agent_feature.view(bs * A, Tm1, -1)

                x_all = self.history_encoder(
                    agent_flat.permute(0, 2, 1).contiguous()
                )

                valid_agent_mask = valid_mask.any(-1).flatten()
                gate    = valid_agent_mask.float().unsqueeze(-1)
                x_agent = (x_all * gate).view(bs, A, self.dim)

                if not self.use_ego_history:
                    ego_feature = data["current_state"][:, : self.state_channel]
                    x_ego       = self.ego_state_emb(ego_feature)
                    x_agent     = torch.cat([x_ego.unsqueeze(1), x_agent[:, 1:]], dim=1)

                x_type   = self.type_emb(category)
                x_agent += x_type
                return x_agent

            module.forward = types.MethodType(_ae_fwd, module)
            patched += 1

    print(f"patch_agent_encoder_for_onnx: patched {patched} AgentEncoder modules")


def patch_planning_model_for_onnx(model: nn.Module) -> None:
    """
    PlanningModel.forward uses torch.atan2 (not in ONNX opset ≤14).
    Replace with an opset-9-compatible approximation.
    """
    import types
    from src.models.planTF.planning_model import PlanningModel

    def _atan2_onnx(y, x):
        pi     = torch.tensor(3.141592653589793, dtype=y.dtype, device=y.device)
        eps    = torch.tensor(1e-7,              dtype=y.dtype, device=y.device)
        sign_x = torch.sign(x + eps)
        offset = torch.where(x >= 0, torch.zeros_like(y),
                             torch.where(y >= 0, pi, -pi))
        return torch.atan(y / (x.abs() + eps)) * sign_x + offset

    patched = 0
    for module in model.modules():
        if isinstance(module, PlanningModel):

            def _pm_fwd(self, data, _atan2=_atan2_onnx):
                agent_pos     = data["agent"]["position"][:, :, self.history_steps - 1]
                agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
                agent_mask    = data["agent"]["valid_mask"][:, :, :self.history_steps]
                polygon_center = data["map"]["polygon_center"]
                polygon_mask   = data["map"]["valid_mask"]

                bs, A = agent_pos.shape[0:2]

                from src.models.planTF.layers.common_layers import build_mlp  # noqa: F401
                position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
                angle    = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
                pos = torch.cat([
                    position,
                    torch.stack([angle.cos(), angle.sin()], dim=-1),
                ], dim=-1)
                pos_embed = self.pos_emb(pos)

                agent_key_padding   = ~(agent_mask.any(-1))
                polygon_key_padding = ~(polygon_mask.any(-1))
                key_padding_mask    = torch.cat(
                    [agent_key_padding, polygon_key_padding], dim=-1
                )

                x_agent   = self.agent_encoder(data)
                x_polygon = self.map_encoder(data)
                x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

                for blk in self.encoder_blocks:
                    x = blk(x, key_padding_mask=key_padding_mask)
                x = self.norm(x)

                trajectory, probability = self.trajectory_decoder(x[:, 0])
                prediction = self.agent_predictor(x[:, 1:A]).view(
                    bs, -1, self.future_steps, 2
                )

                out = {
                    "trajectory":  trajectory,
                    "probability": probability,
                    "prediction":  prediction,
                }

                if not self.training:
                    best_mode = probability.argmax(dim=-1)
                    output_trajectory = trajectory[torch.arange(bs), best_mode]
                    angle_out = _atan2(
                        output_trajectory[..., 3], output_trajectory[..., 2]
                    )
                    out["output_trajectory"] = torch.cat(
                        [output_trajectory[..., :2], angle_out.unsqueeze(-1)], dim=-1
                    )

                return out

            module.forward = types.MethodType(_pm_fwd, module)
            patched += 1

    print(f"patch_planning_model_for_onnx: patched {patched} PlanningModel modules")


def apply_onnx_export_patches(model: nn.Module, natten: str = "exact") -> None:
    """
    Apply every patch used for ONNX export, in order.

    natten="exact"  : _ExactNA1D (numerically equivalent to NATTEN; default)
    natten="global" : _MHAWrapper (legacy global-attention approximation,
                      ~0.35 m mean position error on the released checkpoint)
    """
    natten_patches = {"exact": patch_natten_exact, "global": patch_natten_for_onnx}
    if natten not in natten_patches:
        raise ValueError(f"natten must be one of {sorted(natten_patches)}, got {natten!r}")
    natten_patches[natten](model)
    patch_mha_for_onnx(model)
    patch_boolean_indexing_for_onnx(model)
    patch_agent_encoder_for_onnx(model)
    patch_planning_model_for_onnx(model)
