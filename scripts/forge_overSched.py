import gradio as gr
import inspect
from modules import scripts
from modules import sd_samplers_common
import modules.shared as shared
from modules.shared import opts
import modules.sd_samplers_kdiffusion as K
import k_diffusion.sampling
import modules.sd_samplers
import modules.sd_samplers_extra
import modules.sd_samplers_lcm
import modules.sd_samplers_timesteps
import torch, math
##import pickle

import extensions.Euler_Smea_Dyn_Sampler.smea_sampling as EulerDy

from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup

class patchedKDiffusionSampler(sd_samplers_common.Sampler):

    samplers_list = [
        ('None',        None,                                           {}                                                                              ),
        ('DPM++ 2M',    k_diffusion.sampling.sample_dpmpp_2m,           {}                                                                              ),
        ('Euler a',     k_diffusion.sampling.sample_euler_ancestral,    {"uses_ensd": True}                                                             ),
        ('Euler',       k_diffusion.sampling.sample_euler,              {}                                                                              ),
        ('LMS',         k_diffusion.sampling.sample_lms,                {}                                                                              ),
        ('Heun',        k_diffusion.sampling.sample_heun,               {"second_order": True}                                                          ),
        ('DPM2',        k_diffusion.sampling.sample_dpm_2,              {'discard_next_to_last_sigma': True, "second_order": True}                      ),
        ('DPM2 a',      k_diffusion.sampling.sample_dpm_2_ancestral,    {'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}   ),
        ('DPM++ 2S a',  k_diffusion.sampling.sample_dpmpp_2s_ancestral, {"uses_ensd": True, "second_order": True}                                       ),
        ('DPM++ SDE',   k_diffusion.sampling.sample_dpmpp_sde,          {"second_order": True, "brownian_noise": True}                                  ),
        ('DPM++ 2M SDE',k_diffusion.sampling.sample_dpmpp_2m_sde,       {"brownian_noise": True}                                                        ),
        ('DPM++ 3M SDE',k_diffusion.sampling.sample_dpmpp_3m_sde,       {'discard_next_to_last_sigma': True, "brownian_noise": True}                    ),
        ('DPM fast',    k_diffusion.sampling.sample_dpm_fast,           {"uses_ensd": True}                                                             ),
        ('DPM adaptive',k_diffusion.sampling.sample_dpm_adaptive,       {"uses_ensd": True}                                                             ),
        ('Restart',     modules.sd_samplers_extra.restart_sampler,      {"second_order": True}                                                          ),
        ('LCM',         modules.sd_samplers_lcm.sample_lcm,             {}                                                                              ),
        ('Euler Dy',        EulerDy.sample_euler_dy,                    {}                                                                              ),
        ('Euler SMEA Dy',   EulerDy.sample_euler_smea_dy,               {}                                                                              ),
        #19

#        ('DDIM',        modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.ddim,  {}                                                              ),
#        ('PLMS',        modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.plms,  {}                                                              ),
#        ('UniPC',       modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.unipc, {}                                                              ),
    ]
    
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname)

        self.extra_params = sampler_extra_params.get(funcname, [])

        self.options = options or {}
        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, self.funcname)

        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap = self.model_wrap_cfg.inner_model



    def get_sigmas_LCM(n, sigmas, sigma_min, sigma_max, device='cpu'):
        ##  input sigmas are LCM sigmas calculated to change point, then appended with zero


        listSigmas = sigmas.tolist()
        newSigmas = []

        totalLCMsigmas = len(sigmas) - 2                    #   should be -1, but want higher sigmas. currently generating +1 LCM sigmas, so this is correct
        remainingSteps = n - totalLCMsigmas
        lastLCMsigma = listSigmas[totalLCMsigmas-1]
        delta = (lastLCMsigma - 0.0292) / remainingSteps    #   sigma_min changed by LCM stuff?

        i = 0
        while i < n:
            if i < totalLCMsigmas:
                newSigmas.append(listSigmas[i])
            else:
                lastLCMsigma -= delta
                newSigmas.append(lastLCMsigma)

            i += 1
        newSigmas.append(0.0)

        return torch.tensor(newSigmas, device='cuda:0')


    def get_sigmas_exponential_thresholded(n, sigma_min, sigma_max, device='cpu'):
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
        siglin = torch.linspace(sigma_max/2.71, sigma_min, n, device=device)
        sigmas = torch.max(sigmas, siglin)
        return torch.cat([sigmas, sigmas.new_zeros([1])])

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

    def get_sigmas_fourth(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(1, 0, n, device=device)
        sigmas = sigmas**4
        siglin = torch.linspace(1, 0, n, device=device)
        siglin /= 0.25 * n
        sigmas = torch.max(sigmas, siglin)
        sigmas *= sigma_max - sigma_min
        sigmas += sigma_min
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_4xlinear(n, sigma_min, sigma_max, device='cpu'):
        dropRate1 = 0.75#825
        dropRate2 = 0.75#55
        dropRate3 = 0.01
        b1start = 0
        b2start = n/4
        b3start = n/2
        b4start = 3*n/4

        b1_v = 1.0
        b2_v = b1_v * (1.0 - dropRate1)
        b3_v = b2_v * (1.0 - dropRate2)
        b4_v = b3_v * (1.0 - dropRate3)

        sigmaList = []

        i = 0
        br = b2start - b1start
        while i < br:
            r = b2_v + (b1_v - b2_v) * (br - i) / (br)
            sigmaList.append(r)
            i += 1

        i = 0
        br = b3start - b2start
        while i < br:
            r = b3_v + (b2_v - b3_v) * (br - i) / (br)
            sigmaList.append(r)
            i += 1

        i = 0
        br = b4start - b3start
        while i < br:
            r = b4_v + (b3_v - b4_v) * (br - i) / (br)
            sigmaList.append(r)
            i += 1

        i = 0
        br = n - b4start
        while i < br:
            r = b4_v * (br - (i + 1)) / (br)
            sigmaList.append(r)
            i += 1

        sigmas = torch.tensor(sigmaList, device=device)
        sigmas *= (sigma_max - sigma_min)
        sigmas += sigma_min

        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_geometric(n, sigma_min, sigma_max, device='cpu'):
        #this is same as exponential
        K = (sigma_min / sigma_max)**(1/(n-1))

        res = sigma_max
        sigmaList = [res]
        i = 1
        while (i < n/2):    #best switch point?
            res *= K
            sigmaList.append(res)
            i += 1

        #   switch to linear at mid point - maybe too late?
        #   similar idea to expo_thresholded - maybe tweak switch point
        # possibly some value to this, certainly value to idea of preventing exponential drop off after some point
        #stopping one step early?
        delta = (res - sigma_min) / (n - i)
        while (i < n):
            res -= delta             #last step: i = n-1
            sigmaList.append(res)
            i += 1
            

        sigmas = torch.tensor(sigmaList, device=device)
        print(sigmas)
        return torch.cat([sigmas, sigmas.new_zeros([1])])
       


    def get_sigmas_4xnonlinear(n, sigma_min, sigma_max, device='cpu'):
        target1 = 0.32
        target2 = 0.08
        target3 = target2
        b1start = 0
        b2start = int(n/4)
        b3start = int(n/2)
        b4start = int(3*n/4)

        sigmaList = []

#low ** (1/(n-1))  -> n steps from 1.0 to low
#sigmax * k**n = sigmin
        #log sigmax * kn = log sigmin
        K=(sigmin/sigmax)**(1/n)
        
        i = 0
        r = 1.0
        br = b4start - b3start
        scale = target1 ** (1/(br-1))
        while i < br:
            sigmaList.append(r)
            r *= scale
            i += 1

        i = 0
        br = b3start - b2start
        scale = (target1 - target2) ** (1/(br-1))
        while i < br:
            sigmaList.append(r)
            r *= scale
            i += 1

        i = 0
        br = b4start - b3start
        while i < br:
            sigmaList.append(r)
            i += 1

        i = 0
        br = n - b4start
        b4v = r
        while i < br:
            r = b4v * (br -(i + 1)) / (br)
            sigmaList.append(r)
            i += 1

        sigmas = torch.tensor(sigmaList, device=device)
        sigmas *= (sigma_max - sigma_min)
        sigmas += sigma_min
        print(sigmas)
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

        K.KDiffusionSampler.get_sigmas = OverSchedForge.get_sigmas_backup

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
        elif self.config is not None and OverSchedForge.scheduler == 'exponential thresholded':
            sigmas = patchedKDiffusionSampler.get_sigmas_exponential_thresholded(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'polyexponential':
            sigmas = k_diffusion.sampling.get_sigmas_polyexponential(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'phi':
            sigmas = patchedKDiffusionSampler.get_sigmas_phi(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'fibonacci':
            sigmas = patchedKDiffusionSampler.get_sigmas_fibonacci(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'continuous VP':
            sigmas = k_diffusion.sampling.get_sigmas_vp(n=steps, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == '4th power thresholded':
            sigmas = patchedKDiffusionSampler.get_sigmas_fourth(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
##        elif self.config is not None and OverSchedForge.scheduler == '4x linear':
##            sigmas = patchedKDiffusionSampler.get_sigmas_geometric(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'LCM to linear':
            sigmas = self.model_wrap.get_sigmas(1 + int(OverSchedForge.step * steps))
            sigmas = patchedKDiffusionSampler.get_sigmas_LCM(n=steps, sigmas=sigmas, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        elif self.config is not None and OverSchedForge.scheduler == 'custom':
            sigmas = patchedKDiffusionSampler.get_sigmas_custom(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas


#also need to modify sample_img2img ? seems likely, but get this fully functional first


#find where called from, how is 'p' set? must contain information related to sampler function
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):

#   restore original function immediately, in case of failure later means the main extension can't remove it
        K.KDiffusionSampler.sample = OverSchedForge.sample_backup

        

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas = self.model_wrap.sigmas.to(x.device)

        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps).to(x.device)


        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]


        extra_params_kwargs = self.initialize(p)
        

#p is modules.processing.StableDiffusionProcessingTxt2Img object
        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        listSigmas = sigmas.tolist()

        stepToChange = int(OverSchedForge.step * len(sigmas))

        s1 = torch.tensor(listSigmas[0:stepToChange+1], device='cuda:0')             #   need sigma+1, and makes progress bar count right
        s2 = torch.tensor(listSigmas[stepToChange:len(sigmas)], device='cuda:0')

        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters: 
            extra_params_kwargs['n'] = steps    #   ???, actual number I want with this sampler, or total number?

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[-1].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[0].item()

        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = s1

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args,
                                                                disable=False, callback=self.callback_state, **extra_params_kwargs))


### this is correct, but is it complete??
#self.sampler_extra_args might need updating
#some samplers might need more arguments to be set
#but how useful are they anyway?

        #euler dy slightly different results with switch


        samplerIndex = OverSchedForge.samplerIndex
        self.func = patchedKDiffusionSampler.samplers_list[samplerIndex][1]
        extraParams = patchedKDiffusionSampler.samplers_list[samplerIndex][2]


        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters:
            extra_params_kwargs['n'] = steps

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[-1].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[0].item()

        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = s2
        else:
            extra_params_kwargs.pop('sigmas', None)

        if extraParams.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler
            extra_params_kwargs['s_noise'] = 1.0
            extra_params_kwargs['eta'] = 1.0
        else:
            extra_params_kwargs.pop('eta', None)
            extra_params_kwargs.pop('s_noise', None)
            extra_params_kwargs.pop('noise_sampler', None)

        if extraParams.get('solver_type', None) == 'heun':                  #    what's this 2nd param to get() ??
            extra_params_kwargs['solver_type'] = 'heun'

#maybe should control this by sampler, but seems like they have default values anyway
#add UI for these values?? overkill


        if samplerIndex == 3 or samplerIndex == 5 or samplerIndex == 6 :     #euler, heun, dpm2
            extra_params_kwargs['s_churn'] = shared.opts.s_churn
            extra_params_kwargs['s_tmin'] = shared.opts.s_tmin
            extra_params_kwargs['s_tmax'] = shared.opts.s_tmax
        elif samplerIndex == 17 or samplerIndex == 18:     #euler dy *2
            extra_params_kwargs['s_churn'] = shared.opts.s_churn
            extra_params_kwargs['s_tmin'] = shared.opts.s_tmin
            extra_params_kwargs['s_tmax'] = shared.opts.s_tmax
            extra_params_kwargs['s_noise'] = shared.opts.s_noise
        else:
            extra_params_kwargs.pop('s_churn', None)
            extra_params_kwargs.pop('s_tmin', None)
            extra_params_kwargs.pop('s_tmax', None)


        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, samples, extra_args=self.sampler_extra_args,
                                                                disable=False, callback=self.callback_state, **extra_params_kwargs))

##        #save
##        with open("c:\\temp\\latent.pkl", "wb") as file:
##            pickle.dump(samples, file, pickle.HIGHEST_PROTOCOL)

        self.add_infotext(p)
        sampling_cleanup(unet_patcher)

        return samples











class OverSchedForge(scripts.Script):
    custom = ""
    scheduler = "None"
    samplerIndex = 0
    step = 0.5
    sample_backup = None
    get_sigmas_backup = None

    def __init__(self):
        self.enabled = False
        OverSchedForge.get_sigmas_backup = K.KDiffusionSampler.get_sigmas
        OverSchedForge.sample_backup = K.KDiffusionSampler.sample


    def title(self):
        return "Scheduler Override"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        samplerList = [x[0] for x in patchedKDiffusionSampler.samplers_list]

        with gr.Accordion(open=False, label=self.title()):
            with gr.Row(equalHeight=True):
                enabled = gr.Checkbox(value=False, label='Enabled', scale=0)
                scheduler = gr.Dropdown(["None", "karras", "exponential", "exponential thresholded", "polyexponential", "phi", "fibonacci", "continuous VP", "4th power thresholded", "LCM to linear", "custom"], value="phi", type='value', label='Scheduler choice', scale=0)
                custom = gr.Textbox(value='M * (1-x)**((2-x)*phi) + m * (x)**((2-x)*phi)', max_lines=1, label='custom function', scale=1, visible=False)
            with gr.Row(equalHeight=True):
                sampler = gr.Dropdown(samplerList, value="None", type='index', label='Sampler choice', scale=1)
                step = gr.Slider(minimum=0.01, maximum=0.99, value=0.5, label='Step to change sampler')

            def show_custom(scheduler):
                if scheduler == "custom":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            scheduler.change(
                fn=show_custom,
                inputs=scheduler,
                outputs=custom
            )


        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("os_enabled" in d))),
            (scheduler, "os_scheduler"),
            (custom, "os_custom"),
            (sampler, "os_sampler"),
            (step, "os_step"),
        ]


        return enabled, scheduler, custom, sampler, step



    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, scheduler, custom, sampler, step = script_args
        self.enabled = enabled

        if not enabled:
            return

        OverSchedForge.scheduler = scheduler
        OverSchedForge.custom = custom
        OverSchedForge.samplerIndex = sampler
        OverSchedForge.step = step


        K.KDiffusionSampler.get_sigmas = patchedKDiffusionSampler.get_sigmas

        if sampler != 0:
            K.KDiffusionSampler.sample = patchedKDiffusionSampler.sample


        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(os_enabled = enabled, os_scheduler = scheduler, ))
        if scheduler == "custom":
            params.extra_generation_params.update(dict(os_custom = custom, ))
        params.extra_generation_params.update(dict(os_sampler = patchedKDiffusionSampler.samplers_list[sampler][0], os_step = step, ))

        return


