import torch
import torch.nn as nn
from continual.convit import GPSA, Block as ConVitBlock, ClassAttention
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.patch.timm import make_tome_class


class ToMeGPSA(GPSA):
    def get_attention(self, x, size):
        '''
        copy from convit and modify to match timm
        '''
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        # print(f'rel_indices in get_attention: {self.rel_indices.shape}')
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        # Jordan 这里与convit原始的Attention不同
        if size is not None:
            patch_score = patch_score + size.log()[:, None, None, :, 0]

        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        # print(f'patch_score shape: {patch_score.shape}')
        # print(f'pos_score shape: {pos_score.shape}')
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)

        # 增加k.mean作为返回
        return attn, k.mean(1)

    def forward(self, x, size):
        '''
        modify based on GPSA forward, and apply tome attention changes
        '''
        # Jordan PyTorch中Module的forward方法是自动调用的，也就是在类被初始化后，直接使用名为对象的函数即可触发，详见https://blog.csdn.net/dss_dssssd/article/details/82977170
        # Jordan PyTorch代码Module中定义了__call__方法，调用了forward方法
        # print(f'forward in ToMeGPSA called')
        B, N, C = x.shape
        # print(f'B, N, C: {B}, {N}, {C}')
        # print(f'self: {self}')
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        # modify get_attention to add size and return k.mean
        attn, kmean = self.get_attention(x, size)
        # Jordan reshape将矩阵进行维度变化
        # Jordan permute将矩阵进行转置，将四维矩阵的顺序改为0,2,1,3，也就是将中间两维进行互换
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # add kmean to return
        return x, attn, v, kmean


class ToMeBlock(ConVitBlock):
    # def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=ToMeGPSA, fc=nn.Linear, **kwargs):
    #     print(f'attention_type in ToMeBlock init: {attention_type}')
    #     super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, attention_type, fc, **kwargs)
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x, mask_heads=None, task_index=1, attn_mask=None):
        # print(f'forward of ToMeBlock called')
        # TODO 实现ClassAttention的ToMe后再改下面的代码
        # if isinstance(self.attn, ClassAttention) or isinstance(self.attn, JointCA):  # Like in CaiT
        #     cls_token = x[:, :task_index]

        #     xx = self.norm1(x)
        #     xx, attn, v = self.attn(
        #         xx,
        #         mask_heads=mask_heads,
        #         nb=task_index,
        #         attn_mask=attn_mask
        #     )

        #     cls_token = self.drop_path(xx[:, :task_index]) + cls_token
        #     cls_token = self.drop_path(self.mlp(self.norm2(cls_token))) + cls_token

        #     return cls_token, attn, v

        # print(f'x.shape in TMB foward: {x.shape}')
        xx = self.norm1(x)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        # import inspect
        # print(f'self.attn: {self.attn}')
        # print(f'attn: {inspect.getsource(self.attn)}')
        # self.attn(xx, attn_size)
        # print(f'xx, attn, v, metric = self.attn(xx, attn_size)')
        xx, attn, v, metric = self.attn(xx, attn_size)

        # 这里和标准的ViT不同，而且是attention与MLP，可能会影响ToMe，因为ToMe就是加在Attention和MLP之间的
        x = self._drop_path1(xx) + x

        if 'r' in self._tome_info and len(self._tome_info['r']) > 0:
            # TODO 在test时，r列表长度为0，需要分析为什么，并解决
            # print(f'self._tome_info: {self._tome_info}')
            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        else:
            pass
            # print(f'self._tome_info: {self._tome_info}')

        x = self._drop_path2(self.mlp(self.norm2(x))) + x

        # print(f'x.shape in TMB foward end: {x.shape}')

        return x, attn, v


class ToMeClasstionAttention(ClassAttention):
    def forward(self, x, size, mask_heads=None, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))

        # Apply proportional attention
        # Jordan 这里与原始的Attention不同
        # print(f'attn.shape: {attn.shape}')
        if size is not None:
            print(f'size.log().shape: {size.log().shape}')
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn, v, k.mean(1)


def apply_patch(
    model: torch.nn.Module, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    # print(f'model.__class__: {model.__class__}')
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    # print(f'dir(model): {dir(model)}')
   
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": hasattr(model, 'cls_token') and model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    # TODO 需要对这里进行改造，否则tome的token merging无法生效，也就是没有执行bipartite_soft_matching
    # 此处改造工作量较大，因为tome的forward与DyTox不同，block和attention也不同
    for module in model.modules():
        # print(f'module type: {type(module)}')
        # Jordan 一个Block是由Attention和MLP等组成的
        if isinstance(module, ConVitBlock):
            # print(f'current module is instance of ConVitBlock')
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, GPSA):
            # print(f'current module is instance of GPSA')
            module.__class__ = ToMeGPSA
        elif isinstance(module, ClassAttention):
            module.__class__ = ToMeClasstionAttention