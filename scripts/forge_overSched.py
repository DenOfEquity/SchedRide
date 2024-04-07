import gradio as gr

from modules import scripts
from modules import sd_samplers_common
import modules.shared as shared
from modules.shared import opts
import modules.sd_samplers_kdiffusion as K
import k_diffusion.sampling
import torch, math


class patchedKDiffusionSampler(sd_samplers_common.Sampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname)

        self.extra_params = sampler_extra_params.get(funcname, [])

        self.options = options or {}
        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, self.funcname)

        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap = self.model_wrap_cfg.inner_model

    def get_sigmas_fibonacci (n, sigma_min, sigma_max, device='cpu'):
        revsigmas = torch.linspace(sigma_min, sigma_max, n) #   probably a better way
        sigmas = torch.linspace(0, 1, n) #   probably a better way

        revsigmas[0] *= 1.0
        revsigmas[1] *= 1.0
        i = 2
        while i < n:
            revsigmas[i] = revsigmas[i-2] + revsigmas[i-1]
            i += 1
        i = 0
        while i < n:
            revsigmas[i] /= i+1
            i += 1

        i = 0
        while i < n:
            sigmas[i] = sigma_min + (sigma_max-sigma_min) * ((revsigmas[(n-1)-i] - revsigmas[0]) / revsigmas[n-1])
            i += 1

        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_phi(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = (sigma_max-sigma_min)*((1-x/n)**(phi*phi)) + (sigma_min)*((x/n)**(phi))
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_squared(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(1, 0, n, device=device)
        sigmas = sigmas**2
        sigmas *= sigma_max - sigma_min
        sigmas += sigma_min
        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_linear(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_custom(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)

        phi = (1 + 5**0.5) / 2

        s = 0
        while (s < n):
            x = (s) / (n - 1)
            M = sigma_max
            m = sigma_min

            sigmas[s] = eval((OverSchedForge.custom))   #sigma_max * (1-x)**((x+1)*phi) + sigma_min * (x)**((x+1)*phi)

            s += 1

        return torch.cat([sigmas, sigmas.new_zeros([1])])

        

    def get_sigmas(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        m_sigma_min, m_sigma_max = (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())

        if self.config is not None and OverSchedForge.scheduler == 'karras':
            if opts.use_old_karras_scheduler_sigmas:
                m_sigma_min, m_sigma_max = (0.1, 10)
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'exponential':
            sigmas = k_diffusion.sampling.get_sigmas_exponential(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'polyexponential':
            sigmas = k_diffusion.sampling.get_sigmas_polyexponential(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'phi':
            sigmas = patchedKDiffusionSampler.get_sigmas_phi(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'fibonacci':
            sigmas = patchedKDiffusionSampler.get_sigmas_fibonacci(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'continuous VP':
            sigmas = k_diffusion.sampling.get_sigmas_vp(n=steps, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'squared':
            sigmas = patchedKDiffusionSampler.get_sigmas_squared(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'linear':
            sigmas = patchedKDiffusionSampler.get_sigmas_linear(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'custom':
            sigmas = patchedKDiffusionSampler.get_sigmas_custom(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas



class OverSchedForge(scripts.Script):
    custom = ""
    scheduler = "None"

    def __init__(self):
        self.enabled = False
        self.get_sigmas_backup = None


    def title(self):
        return "Scheduler Override"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label='Enabled', scale=0)
                scheduler = gr.Dropdown(["None", "karras", "exponential", "polyexponential", "phi", "fibonacci", "continuous VP", "squared", "linear", "custom"], value="phi", type='value', label='Scheduler choice', scale=0)
                custom = gr.Textbox(value='M * (1-x)**((2-x)*phi) + m * (x)**((2-x)*phi)', max_lines=1, label='custom function', scale=1)

        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("os_enabled" in d))),
            (scheduler, "os_scheduler"),
            (custom, "os_custom"),
        ]

##        def get_sd_total_steps():
##            if is_img2img:
##                return self.i2i_steps
##            else:
##                return self.t2i_steps

        return enabled, scheduler, custom



    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, scheduler, custom = script_args
        self.enabled = enabled

        if not enabled:
            if self.get_sigmas_backup is not None:
                K.KDiffusionSampler.get_sigmas = self.get_sigmas_backup
            return

        OverSchedForge.scheduler = scheduler
        OverSchedForge.custom = custom

#        print (k_diffusion.k_diffusion_scheduler)
        #   only backup get_sigma if it is differnt
        #   a fail during processing (i.e. bug in extension) means postprocess doesn't get called
        #   so original never restored - need a better way to catch this
        if self.get_sigmas_backup != K.KDiffusionSampler.get_sigmas:
            self.get_sigmas_backup = K.KDiffusionSampler.get_sigmas
        K.KDiffusionSampler.get_sigmas = patchedKDiffusionSampler.get_sigmas


        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(os_enabled = enabled, os_scheduler = scheduler, ))
        if scheduler == "custom":
            params.extra_generation_params.update(dict(os_custom = custom, ))

        return

    def postprocess(self, params, processed, *args):
        if self.enabled and self.get_sigmas_backup is not None:
            K.KDiffusionSampler.get_sigmas = self.get_sigmas_backup
        return

