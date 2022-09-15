import io
import threading
import time

import zerorpc

from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, process_images
import modules.shared as shared
import modules.processing as processing
import modules.sd_samplers as samplers
import modules.memmonitor as memmonitor

sampler_to_index = {s.name.lower(): n for n, s in enumerate(samplers.samplers)}

class SDRPCServer():
    def txt2img(self, opts):
        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=".",
            outpath_grids=".",
            prompt=opts.pop('prompt', ''),
            styles=["", ""],
            negative_prompt=opts.pop('negative_prompt', ''),
            seed=opts.pop('seed', -1),
            subseed=opts.pop('subseed', -1),
            subseed_strength=opts.pop('subseed_strength', 0),
            seed_resize_from_h=opts.pop('seed_resize_from_h', 0),
            seed_resize_from_w=opts.pop('seed_resize_from_w', 0),
            sampler_index=sampler_to_index[opts.pop('sampler', 'lms').lower()],
            batch_size=opts.pop('batch_size', 1),
            n_iter=opts.pop('n_iter', 1),
            steps=opts.pop('steps', 0),
            cfg_scale=opts.pop('cfg_scale', 7.5),
            width=opts.pop('width', 512),
            height=opts.pop('height', 512),
            restore_faces=opts.pop('restore_faces', 0),

            tiling=opts.pop('tiling', 0),
            do_not_save_samples=True,
            do_not_save_grid=True,
        )
        img_format = opts.pop('img_format', 'bmp')
        img_quality = int(opts.pop('img_quality', 85))
        if img_format not in ('bmp', 'png', 'jpeg', 'webp'):
            raise ValueError("unknown img_format %s" % img_format)
        if opts:
            raise ValueError("unhandled opts: " + ' '.join(sorted(opts.keys())))

        start = time.time()

        monitor = memmonitor.MemUsageMonitor()
        try:
            monitor.run()
            processed = process_images(p)
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
        ret.pop('info')

        return ret


def start_server():
    s = zerorpc.Server(SDRPCServer())
    connstr = f"tcp://0.0.0.0:{shared.cmd_opts.port-1}"
    s.bind(connstr)
    print("starting zerorpc server", connstr)
    s.run()

def start():
    threading.Thread(target=start_server).start()
