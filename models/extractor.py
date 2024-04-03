import torch
import torch.nn.modules.utils as nn_utils
import types
from torch import nn
import math


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor(nn.Module):
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, stride, device):
        super().__init__()
        if "v2" in model_name:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model_name = model_name
        self.stride = stride
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        self.set_overlapping_patches()
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self.n_layers = self.get_n_layers()
        self._init_hooks_data()
        
    def set_overlapping_patches(self):
        patch_size = self.get_patch_size()
        if patch_size == self.stride:
            return

        stride = nn_utils._pair(self.stride)
        # assert all([(patch_size // s_) * s_ == patch_size for s_ in
        #             stride]), f'stride {stride} should divide patch_size {patch_size}'
        
        # fix the stride
        self.model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        self.model.interpolate_pos_encoding = types.MethodType(VitExtractor._fix_pos_enc(patch_size, stride), self.model)

        return 0
    
    @staticmethod
    def _fix_pos_enc(patch_size, stride_hw):
        def interpolate_pos_encoding(self, x, w, h):
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.ATTN_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.QKV_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = list(range(self.n_layers))
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img, layers):  # List([B, N, D])
        # if "v2" in self.model_name and layer == self.n_layers - 1:
        #     feature = self.model.forward_features(input_img)["x_prenorm"]
        #     return feature

        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        features = [feature[layer_num] for layer_num in layers]
        # features = torch.cat(features, dim=2)
        features = torch.stack(features).mean(dim=0)
        return features

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 14

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        return 1 + (w - self.get_patch_size()) // self.stride

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        return 1 + (h - self.get_patch_size()) // self.stride

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num
    
    def get_n_layers(self):
        if "s" in self.model_name:
            return 12
        elif "b" in self.model_name:
            return 12
        elif "l" in self.model_name:
            return 24
        elif "g" in self.model_name:
            return 40
        else:
            raise Exception("invalid model name")

    def get_head_num(self):
        if "s" in self.model_name:
            return 6
        elif "b" in self.model_name:
            return 12
        elif "l" in self.model_name:
            return 16
        elif "g" in self.model_name:
            return 24   
        else:
            raise Exception("invalid model name")

    def get_embedding_dim(self):
        return VitExtractor.get_embedding_dim(self.model_name)
            
    @staticmethod
    def get_embedding_dim(model_name):
        if "dino" in model_name:
            if "s" in model_name:
                return 384
            elif "b" in model_name:
                return 768
            elif "l" in model_name:
                return 1024
            elif "g" in model_name:
                return 1536
            else:
                raise Exception("invalid model name")

    def get_queries_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        q = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 0, :]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        k = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 1, :]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        v = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 2, :]
        return v

    def get_keys_from_input(self, input_img, layers):
        keys = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            keys.append(self.get_keys_from_qkv(qkv_features, input_img.shape))
        keys = torch.cat(keys, dim=2)
        return keys
    
    def get_queries_from_input(self, input_img, layers):
        q = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            q.append(self.get_queries_from_qkv(qkv_features, input_img.shape)) # B x (HxW+1) x C
        q = torch.cat(q, dim=2)
        return q
    
    def get_values_from_input(self, input_img, layers):
        v = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            v.append(self.get_values_from_qkv(qkv_features, input_img.shape))
        v = torch.cat(v, dim=2)
        return v

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layers=[layer_num])
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map
