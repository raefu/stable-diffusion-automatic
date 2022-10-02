# Author: Therefore Games
# https://github.com/ThereforeGames/txt2mask

from typing import final
import modules.scripts as scripts
import gradio as gr

from modules import processing

import torch
import cv2
import requests
import os.path

# from https://github.com/timojl/clipseg
# MIT License
#
# This license does not apply to the model weights.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from PIL import ImageChops, Image, ImageOps, ImageFilter
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy
import math
from os.path import basename, dirname, join, isfile
import torch
from torch import nn
from torch.nn import functional as nnf


def get_prompt_list(prompt):
    if prompt == 'plain':
        return ['{}']
    elif prompt == 'fixed':
        return ['a photo of a {}.']
    elif prompt == 'shuffle':
        return [
            'a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.',
            '{}.'
        ]
    elif prompt == 'shuffle+':
        return [
            'a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.',
            '{}.', 'a cropped photo of a {}.', 'a good photo of a {}.',
            'a photo of one {}.', 'a bad photo of a {}.', 'a photo of the {}.'
        ]
    else:
        raise ValueError('Invalid value for prompt')


def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses).
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module.
    """

    x_ = b.ln_1(x)
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight,
                         b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim)**-0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads,
                            b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads,
                            b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads,
                            b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(
        1, 2))  #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:

        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)

        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0,
                                1:] = attn_output_weights[:, 0,
                                                          1:] * attn_mask[None,
                                                                          ...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:,
                                1:] = attn_output_weights[:, 1:,
                                                          1:] * attn_mask[:,
                                                                          None]

    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


class CLIPDenseBase(nn.Module):

    def __init__(self, version, reduce_cond, reduce_dim, prompt, n_tokens):
        super().__init__()

        import clip

        # prec = torch.FloatTensor
        self.clip_model, _ = clip.load(version, device='cpu', jit=False)
        self.model = self.clip_model.visual

        # if not None, scale conv weights such that we obtain n_tokens.
        self.n_tokens = n_tokens

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # conditional
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None

        self.film_mul = nn.Linear(512 if reduce_cond is None else reduce_cond,
                                  reduce_dim)
        self.film_add = nn.Linear(512 if reduce_cond is None else reduce_cond,
                                  reduce_dim)

        self.reduce = nn.Linear(768, reduce_dim)

        self.prompt_list = get_prompt_list(prompt)

        # precomputed prompts
        import pickle
        if isfile('precomputed_prompt_vectors.pickle'):
            precomp = pickle.load(
                open('precomputed_prompt_vectors.pickle', 'rb'))
            self.precomputed_prompts = {
                k: torch.from_numpy(v)
                for k, v in precomp.items()
            }
        else:
            self.precomputed_prompts = dict()

    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2

        a = self.model.positional_embedding[1:].T.view(1, 768,
                                                       *self.token_shape)
        b = nnf.interpolate(a, new_size, mode='bicubic',
                            align_corners=False).squeeze(0).view(
                                768, new_size[0] * new_size[1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):

        with torch.no_grad():

            inp_size = x_inp.shape[2:]

            if self.n_tokens is not None:
                stride2 = x_inp.shape[2] // self.n_tokens
                conv_weight2 = nnf.interpolate(self.model.conv1.weight,
                                               (stride2, stride2),
                                               mode='bilinear',
                                               align_corners=True)
                x = nnf.conv2d(x_inp,
                               conv_weight2,
                               bias=self.model.conv1.bias,
                               stride=stride2,
                               dilation=self.model.conv1.dilation)
            else:
                x = self.model.conv1(x_inp)  # shape = [*, width, grid, grid]

            x = x.reshape(x.shape[0], x.shape[1],
                          -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            x = torch.cat([
                self.model.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device), x
            ],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]

            standard_n_tokens = 50 if self.model.conv1.kernel_size[
                0] == 32 else 197

            if x.shape[1] != standard_n_tokens:
                new_shape = int(math.sqrt(x.shape[1] - 1))
                x = x + self.rescaled_pos_emb(
                    (new_shape, new_shape)).to(x.dtype)[None, :, :]
            else:
                x = x + self.model.positional_embedding.to(x.dtype)

            x = self.model.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], []
            for i, res_block in enumerate(self.model.transformer.resblocks):

                if mask is not None:
                    mask_layer, mask_type, mask_tensor = mask
                    if mask_layer == i or mask_layer == 'all':
                        # import ipdb; ipdb.set_trace()
                        size = int(math.sqrt(x.shape[0] - 1))

                        attn_mask = (mask_type,
                                     nnf.interpolate(
                                         mask_tensor.unsqueeze(1).float(),
                                         (size, size)).view(
                                             mask_tensor.shape[0],
                                             size * size))

                    else:
                        attn_mask = None
                else:
                    attn_mask = None

                x, aff_per_head = forward_multihead_attention(
                    x, res_block, with_aff=True, attn_mask=attn_mask)

                if i in extract_layers:
                    affinities += [aff_per_head]

                    #if self.n_tokens is not None:
                    #    activations += [nnf.interpolate(x, inp_size, mode='bilinear', align_corners=True)]
                    #else:
                    activations += [x]

                if len(extract_layers) > 0 and i == max(
                        extract_layers) and skip:
                    print('early skip')
                    break

            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_post(x[:, 0, :])

            if self.model.proj is not None:
                x = x @ self.model.proj

            return x, activations, affinities

    def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {
                list, tuple
        } and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(
                conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond

    def compute_conditional(self, conditional):
        import clip

        dev = next(self.parameters()).device

        if type(conditional) in {list, tuple}:
            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
        else:
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]

        if self.shift_vector is not None:
            return cond + self.shift_vector
        else:
            return cond


class CLIPDensePredT(CLIPDenseBase):

    def __init__(self,
                 version='ViT-B/32',
                 extract_layers=(3, 6, 9),
                 cond_layer=0,
                 reduce_dim=128,
                 n_heads=4,
                 prompt='fixed',
                 extra_blocks=0,
                 reduce_cond=None,
                 fix_shift=False,
                 learn_trans_conv_only=False,
                 limit_to_clip_only=False,
                 upsample=False,
                 add_calibration=False,
                 rev_activations=False,
                 trans_conv=None,
                 n_tokens=None,
                 complex_trans_conv=False):

        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)
        # device = 'cpu'

        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.process_cond = None
        self.rev_activations = rev_activations

        depth = len(extract_layers)

        if add_calibration:
            self.calibration_conds = 1

        self.upsample_proj = nn.Conv2d(reduce_dim, 1,
                                       kernel_size=1) if upsample else None

        self.add_activation1 = True

        self.version = version

        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14)}[version]

        if fix_shift:
            # self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'clip_text_shift_vector.pth')), requires_grad=False)
            self.shift_vector = nn.Parameter(torch.load(
                join(dirname(basename(__file__)), 'shift_text_to_vis.pth')),
                                             requires_grad=False)
            # self.shift_vector = nn.Parameter(-1*torch.load(join(dirname(basename(__file__)), 'shift2.pth')), requires_grad=False)
        else:
            self.shift_vector = None

        if trans_conv is None:
            trans_conv_ks = {
                'ViT-B/32': (32, 32),
                'ViT-B/16': (16, 16)
            }[version]
        else:
            # explicitly define transposed conv kernel size
            trans_conv_ks = (trans_conv, trans_conv)

        if not complex_trans_conv:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim,
                                                 1,
                                                 trans_conv_ks,
                                                 stride=trans_conv_ks)
        else:
            assert trans_conv_ks[0] == trans_conv_ks[1]

            tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)

            self.trans_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim,
                                   reduce_dim // 2,
                                   kernel_size=tp_kernels[0],
                                   stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2,
                                   1,
                                   kernel_size=tp_kernels[1],
                                   stride=tp_kernels[1]),
            )


#        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)

        assert len(self.extract_layers) == depth

        self.reduces = nn.ModuleList(
            [nn.Linear(768, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads)
            for _ in range(len(self.extract_layers))
        ])
        self.extra_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads)
            for _ in range(extra_blocks)
        ])

        # refinement and trans conv

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)

            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        self.prompt_list = get_prompt_list(prompt)

    def forward(self,
                inp_image,
                conditional=None,
                return_features=False,
                mask=None):

        assert type(return_features) == bool

        inp_image = inp_image.to(self.model.positional_embedding.device)

        if mask is not None:
            raise ValueError('mask not supported')

        # x_inp = normalize(inp_image)
        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device

        cond = self.get_cond_vec(conditional, bs)

        visual_q, activations, _ = self.visual_forward(
            x_inp, extract_layers=[0] + list(self.extract_layers))

        activation1 = activations[0]
        activations = activations[1:]

        _activations = activations[::
                                   -1] if not self.rev_activations else activations

        a = None
        for i, (activation, block, reduce) in enumerate(
                zip(_activations, self.blocks, self.reduces)):

            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            if i == self.cond_layer:
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond)

                a = self.film_mul(cond) * a + self.film_add(cond)

            a = block(a)

        for block in self.extra_blocks:
            a = a + block(a)

        a = a[1:].permute(1, 2, 0)  # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size)

        a = self.trans_conv(a)

        if self.n_tokens is not None:
            a = nnf.interpolate(a,
                                x_inp.shape[2:],
                                mode='bilinear',
                                align_corners=True)

        if self.upsample_proj is not None:
            a = self.upsample_proj(a)
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear')

        if return_features:
            return a, visual_q, cond, [activation1] + activations
        else:
            return a,


class CLIPDensePredTMasked(CLIPDensePredT):

    def __init__(self,
                 version='ViT-B/32',
                 extract_layers=(3, 6, 9),
                 cond_layer=0,
                 reduce_dim=128,
                 n_heads=4,
                 prompt='fixed',
                 extra_blocks=0,
                 reduce_cond=None,
                 fix_shift=False,
                 learn_trans_conv_only=False,
                 refine=None,
                 limit_to_clip_only=False,
                 upsample=False,
                 add_calibration=False,
                 n_tokens=None):

        super().__init__(version=version,
                         extract_layers=extract_layers,
                         cond_layer=cond_layer,
                         reduce_dim=reduce_dim,
                         n_heads=n_heads,
                         prompt=prompt,
                         extra_blocks=extra_blocks,
                         reduce_cond=reduce_cond,
                         fix_shift=fix_shift,
                         learn_trans_conv_only=learn_trans_conv_only,
                         limit_to_clip_only=limit_to_clip_only,
                         upsample=upsample,
                         add_calibration=add_calibration,
                         n_tokens=n_tokens)

    def visual_forward_masked(self, img_s, seg_s):
        return super().visual_forward(img_s, mask=('all', 'cls_token', seg_s))

    def forward(self, img_q, cond_or_img_s, seg_s=None, return_features=False):

        if seg_s is None:
            cond = cond_or_img_s
        else:
            img_s = cond_or_img_s

            with torch.no_grad():
                cond, _, _ = self.visual_forward_masked(img_s, seg_s)

        return super().forward(img_q, cond, return_features=return_features)

debug = False

model = None

def init_model():
    global model

    if model is not None:
        return

    print("Loading weights for ClipSeg model...")

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model_dir = "./repositories/clipseg/weights"
    os.makedirs(model_dir, exist_ok=True)
    d64_file = f"{model_dir}/rd64-uni-refined.pth"

    def download_file(filename, url):
        with open(filename, 'wb') as fout:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            # Write response data to file
            for block in response.iter_content(4096):
                fout.write(block)

    # Download model weights if we don't have them yet
    if not os.path.exists(d64_file):
        print("Downloading clipseg model weights...")
        download_file(
            d64_file,
            "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth"
        )
        # Mirror:
        # https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth
        # https://github.com/timojl/clipseg/raw/master/weights/rd16-uni.pth

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(d64_file, map_location=torch.device('cuda')), strict=False)

class Script(scripts.Script):
    def title(self):
        return "txt2mask v0.1.2-raefu"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        mask_prompt = gr.Textbox(label="Mask prompt", lines=1)
        negative_mask_prompt = gr.Textbox(label="Negative mask prompt",
                                          lines=1)
        mask_precision = gr.Slider(label="Mask precision",
                                   minimum=0.0,
                                   maximum=255.0,
                                   step=1.0,
                                   value=100.0)
        mask_padding = gr.Slider(label="Mask padding",
                                 minimum=0.0,
                                 maximum=500.0,
                                 step=1.0,
                                 value=0.0)
        brush_mask_mode = gr.Radio(label="Brush mask mode",
                                   choices=['discard', 'add', 'subtract'],
                                   value='discard',
                                   type="index",
                                   visible=False)
        mask_output = gr.Checkbox(label="Show mask in output?", value=True)

        return [
            mask_prompt, negative_mask_prompt, mask_precision, mask_padding,
            brush_mask_mode, mask_output
        ]

    def run(self, p, mask_prompt, negative_mask_prompt, mask_precision,
            mask_padding, brush_mask_mode, mask_output):
        init_model()

        def overlay_mask_part(img_a, img_b, mode):
            if mode == 0:
                img_a = ImageChops.darker(img_a, img_b)
            else:
                img_a = ImageChops.lighter(img_a, img_b)
            return img_a

        def process_mask_parts(these_preds,
                               these_prompt_parts,
                               mode,
                               final_img=None):
            for i in range(these_prompt_parts):
                arr = torch.sigmoid(these_preds[i][0])

                filename = f"mask_{mode}_{i}.png"
                if debug:
                    plt.imsave(filename, arr)

                arr = (arr.numpy() * 256).astype(numpy.uint8)

                _, bw_image = cv2.threshold(arr, mask_precision, 255, cv2.THRESH_BINARY)

                if mode == 0:
                    bw_image = numpy.invert(bw_image)

                if debug:
                    print(f"bw_image: {bw_image}")
                    print(f"final_img: {final_img}")

                # overlay mask parts
                bw_image = Image.fromarray(cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGBA))
                if i > 0 or final_img is not None:
                    bw_image = overlay_mask_part(bw_image, final_img, mode)

                # For debugging only:
                if debug:
                    bw_image.save(f"processed_{filename}")

                final_img = bw_image

            return final_img

        def get_mask():
            delimiter_string = ","

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ])
            img = transform(p.init_images[0]).unsqueeze(0)

            prompts = mask_prompt.split(delimiter_string)
            prompt_parts = len(prompts)
            negative_prompts = negative_mask_prompt.split(delimiter_string)
            negative_prompt_parts = len(negative_prompts)

            # predict
            with torch.no_grad():
                model.to(torch.device('cuda'))
                preds = model(img.repeat(prompt_parts, 1, 1, 1), prompts)[0]
                negative_preds = model(
                    img.repeat(negative_prompt_parts, 1, 1, 1),
                    negative_prompts)[0]
                model.to(torch.device('cpu'))
                preds = preds.cpu()
                negative_preds = negative_preds.cpu()

            #tests
            if debug:
                print("Check initial mask vars before processing...")
                print(f"p.image_mask: {p.image_mask}")
                print(f"p.latent_mask: {p.latent_mask}")
                print(f"p.mask_for_overlay: {p.mask_for_overlay}")

            if brush_mask_mode == 1 and p.image_mask is not None:
                final_img = p.image_mask.convert("RGBA")
            else:
                final_img = None

            # process masking
            final_img = process_mask_parts(preds, prompt_parts, 1, final_img)

            # process negative masking
            if brush_mask_mode == 2 and p.image_mask is not None:
                p.image_mask = ImageOps.invert(p.image_mask)
                p.image_mask = p.image_mask.convert("RGBA")
                final_img = overlay_mask_part(final_img, p.image_mask, 0)
            if negative_mask_prompt:
                final_img = process_mask_parts(negative_preds,
                                               negative_prompt_parts, 0,
                                               final_img)

            # Increase mask size with blur + threshold
            if mask_padding > 0:
                blur = final_img.filter(ImageFilter.GaussianBlur(radius=mask_padding))
                final_img = blur.point(lambda x: 255 * (x > 0))

            return (final_img)

        # Set up processor parameters correctly
        p.mode = 1
        p.mask_mode = 1
        p.image_mask = get_mask().resize(
            (p.init_images[0].width, p.init_images[0].height))
        p.mask_for_overlay = p.image_mask
        p.latent_mask = None  # fixes inpainting full resolution

        processed = processing.process_images(p)

        if mask_output:
            processed.images.append(p.image_mask)

        return processed
