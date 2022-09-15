import io
import inspect
import threading
import time

import zerorpc

from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, process_images
from modules import shared, processing, sd_samplers as samplers, memmonitor, img2img
i2i_argnames = frozenset(inspect.getfullargspec(img2img.img2img)[0])

class SDRPCServer():
    def txt2img(self, opts):
        sampler_to_index = {s.name.lower(): n for n, s in enumerate(samplers.samplers)}
        upscaler_to_index = {s.name.lower(): n for n, s in enumerate(shared.sd_upscalers)}

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
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

        img_format = opts.pop('img_format', 'bmp')
        img_quality = int(opts.pop('img_quality', 85))
        if img_format not in ('bmp', 'png', 'jpeg', 'webp'):
            raise ValueError("unknown img_format %s" % img_format)
        if opts:
            raise ValueError("unhandled opts: " + ' '.join(sorted(opts.keys())))

        if upscale:
            upscale_restore_faces = p.restore_faces
            p.restore_faces = False

        start = time.time()

        monitor = memmonitor.MemUsageMonitor()
        processed2 = None
        try:
            monitor.run()
            processed = process_images(p)
            if upscale:
                # gross way to not have to respecify every arg
                p2 = dict(p.__dict__)
                p2['prompt_style'], p2['prompt_style2'] = p2.pop('styles')
                p2.update(dict(
                    steps=4,
                    mode=2,
                    init_img=processed.images[0],
                    denoising_strength=0.2,
                    upscaler_index=upscaler_to_index['real-esrgan 2x plus'],
                    upscale_overlap=64,
                    restore_faces=upscale_restore_faces,
                ))
                p2 = {k: v for k, v in p2.items() if k in i2i_argnames}
                for k in i2i_argnames:
                    if k not in p2:
                        p2[k] = None
                print(p2)
                upscaleproc = img2img.img2img(**p2)
                processed.images = upscaleproc[0]

        finally:
            monitor.stop()
        shared.total_tqdm.clear()

        ret = dict(processed.__dict__)
        images = []
        for im in ret['images']:
            b = io.BytesIO()
            im.save(b, format=img_format, quality=img_quality)
            images.append(b.getvalue())
        ret['images'] = images
        ret['elapsed'] = time.time() - start
        ret['vram_used'], ret['vram_total'] = monitor.read()
        if '\n\n' in ret['info']:
            ret['warn'] = ret.pop('info').split('\n\n', 1)[1]
        for k in ('prompt', 'info', 'negative_prompt'):
            ret.pop(k, None)

        return ret


def start_server():
    s = zerorpc.Server(SDRPCServer(), heartbeat=None)
    connstr = f"tcp://0.0.0.0:{shared.cmd_opts.port-1}"
    s.bind(connstr)
    print("starting zerorpc server", connstr)
    s.run()

def start():
    threading.Thread(target=start_server).start()
