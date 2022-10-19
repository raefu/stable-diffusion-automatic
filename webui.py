import glob
import os
import threading
import time
import importlib
import signal
import threading

from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork


import modules.rpc

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        devices.torch_gc()

        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.job_timestamp = shared.state.get_job_timestamp()
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.textinfo = None

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        devices.torch_gc()

        return res

    return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)

def initialize():
    if 'R-ESRGAN 2x+' not in shared.opts.realesrgan_enabled_models:
        shared.opts.realesrgan_enabled_models.append('R-ESRGAN 2x+')

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

    print(sorted(modules.sd_models.checkpoints_list.keys()))

    shared.sd_model = modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    shared.opts.sd_checkpoint_cache = max(10, shared.opts.sd_checkpoint_cache)
    if not os.getenv("NO_PRELOAD"):
        # preload wd-v1-2-full-ema-pruned.ckpt
        wd = modules.sd_models.select_checkpoint_by_hash('e393dbe0')
        wd and modules.sd_models.reload_model_weights(shared.sd_model, wd)
        # preload wd-v1-3-float16.ckpt
        wd13 = modules.sd_models.select_checkpoint_by_hash('b6d184f6')
        wd13 and modules.sd_models.reload_model_weights(shared.sd_model, wd13)
        # preload gg1342_testrun1_pruned.ckpt
        gg = modules.sd_models.select_checkpoint_by_hash('13d7b26b')
        gg and modules.sd_models.reload_model_weights(shared.sd_model, gg)


def webui():
    initialize()

    # make the program just exit at ctrl+c without waiting for anything

    rpc = modules.rpc.start(queue_lock)

    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        rpc.stop()  # graceful shutdown
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    if cmd_opts.upnp:
        from modules.upnp import map_local
        threading.Thread(target=map_local, args=(cmd_opts.upnp,), daemon=True).start()

    while 1:
        demo = modules.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)

        app, local_url, share_url = demo.launch(
            share=cmd_opts.share,
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        while 1:
            time.sleep(0.5)
            if getattr(demo, 'do_restart', False):
                time.sleep(0.5)
                demo.close()
                time.sleep(0.5)
                break

        sd_samplers.set_samplers()

        print('Reloading Custom Scripts')
        modules.scripts.reload_scripts(os.path.join(script_path, "scripts"))
        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Refreshing Model List')
        modules.sd_models.list_models()
        print('Restarting Gradio')


if __name__ == "__main__":
    webui()
