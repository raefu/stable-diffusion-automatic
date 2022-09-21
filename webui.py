import glob
import os
import threading

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.ldsr_model as ldsr
import modules.extras
import modules.lowvram
import modules.txt2img
import modules.img2img
import modules.swinir as swinir
import modules.sd_models

import modules.rpc

modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
swinir.load_models(cmd_opts.swinir_models_path)
realesrgan.setup_realesrgan()
ldsr.add_lsdr()
queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))

try:
    # preload wd-v1-2-full-ema-pruned.ckpt
    modules.sd_models.load_model([c for c in modules.sd_models.checkpoints_list.values() if c.hash == 'e393dbe0'][0])
except IndexError:
    pass

def webui():
    # make the program just exit at ctrl+c without waiting for anything

    rpc = modules.rpc.start(queue_lock)

    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        rpc.stop()  # graceful shutdown
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(modules.extras.run_extras),
        run_pnginfo=modules.extras.run_pnginfo
    )


    demo.launch(
        share=cmd_opts.share,
        server_name="0.0.0.0" if cmd_opts.listen else None,
        server_port=cmd_opts.port,
        debug=cmd_opts.gradio_debug,
        auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
        inbrowser=cmd_opts.autolaunch,
    )


if __name__ == "__main__":
    webui()
