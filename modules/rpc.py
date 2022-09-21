import contextlib
import io
import inspect
import os
import subprocess
import threading
import time

import torch

import modules.sd_hijack as sd_hijack
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, process_images
from modules import shared, processing, sd_samplers as samplers, memmonitor, img2img, devices, sd_models

from modules.pogorpc import PogoServer


class SDRPCServer:
    def __init__(self, queue_lock):
        i2i_sig = inspect.getfullargspec(img2img.img2img)
        self.i2i_argnames = frozenset(i2i_sig.args + i2i_sig.kwonlyargs)
        self.queue_lock = queue_lock

    def ping(self, n):
        return n + 1

    def caps(self, _):
        try:
            rev = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf8').strip()
        except subprocess.CalledProcessError:
            rev = None

        device_name = 'unknown'
        device_total_ram = 0
        device_used_ram = 0

        try:
            dev = torch.cuda.get_device_properties(0)
            device_name = dev.name
            device_total_ram = dev.total_memory
            device_used_ram = torch.cuda.memory_reserved(0)
        except AssertionError:
            pass

        return {
            'git-rev': rev,
            'device': device_name,
            'vram_total': round(device_total_ram / 1024 ** 3, 2),
            'vram_used': round(device_used_ram / 1024 ** 3, 2),
            'checkpoints': sorted(set(c.hash for c in sd_models.checkpoints_list.values())),
            'upscalers': sorted(set(s.name.lower() for s in shared.sd_upscalers)),
            'face_restorers': sorted(set(s.name().lower() for s in shared.face_restorers)),
        }

    def txts2imgs(self, opts_list):
        ret = []
        with self.queue_lock:  # reentrant so this is okay
            for opts in opts_list:
                ret.append(self.txt2img(opts))
        return ret

    def txt2img(self, opts):
        sampler_to_index = {s.name.lower(): n for n, s in enumerate(samplers.samplers)}
        upscaler_to_index = {s.name.lower(): n for n, s in enumerate(shared.sd_upscalers)}

        model_req = opts.pop('model', 'model')
        model_hash = {
            'model': '44ef7ed9',
            'waifu': 'e393dbe0',
            'gg1342': '13d7b26b',
        }[model_req]
        model_ckpt = None
        for ckpt in sd_models.checkpoints_list.values():
            if ckpt.hash == model_hash:
                model_ckpt = ckpt
                break
        else:
            raise ValueError(f"unknown model '{model_req}' with hash {model_hash}")

        p = StableDiffusionProcessingTxt2Img(
            outpath_samples=".",
            outpath_grids=".",
            prompt=opts.pop('prompt', ''),
            styles=["", ""],
            negative_prompt=opts.pop('prompt_neg', ''),
            seed=opts.pop('seed', -1),
            subseed=opts.pop('subseed', -1),
            subseed_strength=opts.pop('subseed_strength', 0),
            seed_resize_from_h=opts.pop('seed_resize_from_h', 0),
            seed_resize_from_w=opts.pop('seed_resize_from_w', 0),
            sampler_index=sampler_to_index[opts.pop('sampler', 'lms').lower()],
            batch_size=opts.pop('batch_size', 1),
            n_iter=opts.pop('n_iter', 1),
            steps=opts.pop('steps', 0),
            cfg_scale=opts.pop('cfg', 7.5),
            width=opts.pop('width', 512),
            height=opts.pop('height', 512),
            restore_faces=opts.pop('restore_faces', 0),
            tiling=opts.pop('tiling', 0),
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        upscale = opts.pop('upscale', False)
        opt_split_attention = opts.pop('opt1', False)

        img_format = opts.pop('img_format', 'bmp')
        img_quality = int(opts.pop('img_quality', 85))
        if img_format not in ('bmp', 'png', 'jpeg', 'webp'):
            raise ValueError("unknown img_format %s" % img_format)
        if opts:
            raise ValueError("unhandled opts: " + ' '.join(sorted(opts.keys())))

        if upscale:
            upscale_restore_faces = p.restore_faces
            p.restore_faces = False

        monitor = memmonitor.MemUsageMonitor()
        processed2 = None
        try:
            monitor.start()

            with self.queue_lock, (sd_hijack.opt_split_attention() if opt_split_attention else contextlib.nullcontext()):
                start = time.time()
                if shared.sd_model.sd_model_hash != model_ckpt.hash:
                    old_model = shared.sd_model
                    shared.sd_model = sd_models.load_model(model_ckpt)
                    shared.sd_model.to(shared.device)
                    old_model.to(devices.cpu)
                    devices.torch_gc()
                p.sd_model = shared.sd_model
                processed = process_images(p)
                if upscale:
                    # gross way to not have to respecify every arg
                    p2 = dict(p.__dict__)
                    p2['prompt_style'], p2['prompt_style2'] = p2.pop('styles')
                    p2.update(dict(
                        sd_model=shared.sd_model,
                        steps=4,
                        mode=2,
                        init_img=processed.images[0],
                        denoising_strength=0.2,
                        upscaler_index=upscaler_to_index['real-esrgan 2x plus'],
                        upscale_overlap=64,
                        restore_faces=upscale_restore_faces,
                    ))
                    p2 = {k: v for k, v in p2.items() if k in self.i2i_argnames}
                    for k in self.i2i_argnames:
                        if k not in p2:
                            p2[k] = None
                    upscaleproc = img2img.img2img(**p2)
                    processed.images = upscaleproc[0]
                shared.total_tqdm.clear()
        finally:
            monitor.stop()

        ret = dict(processed.__dict__)
        images = []
        for im in ret['images']:
            b = io.BytesIO()
            im.save(b, format=img_format, quality=img_quality)
            images.append(b.getvalue())
        ret['images'] = images
        ret['elapsed'] = round(time.time() - start, 3)
        ret['vram_used'], ret['vram_total'] = monitor.read()
        ret['model_hash'] = model_ckpt.hash

        if not ret['subseed_strength']:
            ret.pop('subseed')
            ret.pop('subseed_strength')
        # remove unnecessary repetition in the response
        for k in ('sampler_index', 'seed_resize_from_w', 'seed_resize_from_h', 'denoising_strength',
            'extra_generation_params', 'index_of_first_image', 'all_prompts', 'all_seeds', 'all_subseeds',
            'face_restoration_model', 'restore_faces', 'batch_size', 'upscale', 'n_iter', 'width', 'height',
            'steps', 'sampler', 'cfg_scale', 'sd_model_hash'):
            ret.pop(k, None)

        if not ret['vram_used']:
            ret.pop('vram_used')
            ret.pop('vram_total')

        if '\n\n' in ret['info']:
            ret['warn'] = ret.pop('info').split('\n\n', 1)[1]
        for k in ('prompt', 'info', 'negative_prompt'):
            ret.pop(k, None)

        return ret

def read_key(name):
    return open(os.path.join('keys', name), 'rb').read()

def start(queue_lock):
    srv = PogoServer(SDRPCServer(queue_lock),
        port=(shared.cmd_opts.port or 7860) - 1,
        creds=(read_key('server.key'), read_key('server.crt'), read_key('ca.crt')))
    threading.Thread(target=srv.run).start()
    return srv
