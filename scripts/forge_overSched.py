import gradio as gr
import inspect
from modules import scripts
import modules.shared as shared
from modules.shared import opts
import modules.sd_samplers_kdiffusion as K
import k_diffusion.sampling
import k_diffusion.external
import modules.sd_samplers
import modules.sd_samplers_common
import modules.sd_samplers_extra
import modules.sd_samplers_lcm
import modules.sd_samplers_timesteps
import torch, math
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.ui_components import ToolButton                                                     


#import extensions.Euler_Smea_Dyn_Sampler.smea_sampling as EulerDy

from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup

class patchedKDiffusionSampler(modules.sd_samplers_common.Sampler):

    import importlib
    EulerDy = importlib.import_module("extensions.Euler-Smea-Dyn-Sampler.smea_sampling")

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
    ]

    #   is there a better way than hardcoding functions?
    #   seems like should be able to extract from modules.sd_samplers somehow
    for i in modules.sd_samplers.all_samplers:
        if i[0] == "Euler Dy":
            samplers_list.extend([
               (i[0],  EulerDy.sample_euler_dy,            {}  ),
            ])
        elif i[0] == "Euler SMEA Dy":
            samplers_list.extend([
                (i[0],  EulerDy.sample_euler_smea_dy,       {}  ),
            ])
        elif i[0] == "Euler Negative":
            samplers_list.extend([
                (i[0],  EulerDy.sample_euler_negative,      {}  ),
            ])
        elif i[0] == "Euler Negative Dy":
            samplers_list.extend([
                (i[0],  EulerDy.sample_euler_dy_negative,   {}  ),
            ])
            break
        #also get function name from all_samplers
        
        #18 next

#        ('DDIM',        modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.ddim,  {}                                                              ),
#        ('PLMS',        modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.plms,  {}                                                              ),
#        ('UniPC',       modules.sd_samplers_timesteps.sd_samplers_timesteps_impl.unipc, {}                                                              ),
    
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
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(n-1))**(phi*phi))
        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_cosine(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(1, 0, n, device=device)

        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C

        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_fourth(n, sigma_min, sigma_max, device='cpu'):
        sigmas = torch.linspace(1, 0, n, device=device)
        sigmas = sigmas**4
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

        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def get_sigmas_custom(n, sigma_min, sigma_max, device='cpu'):
        if isinstance(eval(OverSchedForge.custom), list):
            sigmasList = eval(OverSchedForge.custom)
            xs = np.linspace(0, 1, len(sigmasList))
            ys = np.log(sigmasList[::-1])
            
            new_xs = np.linspace(0, 1, n)
            new_ys = np.interp(new_xs, xs, ys)
            
            interpolated_ys = np.exp(new_ys)[::-1].copy()
            sigmas = torch.tensor(interpolated_ys, device=device)
        else:
            sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)

            phi = (1 + 5**0.5) / 2
            pi = math.pi

            s = 0
            while (s < n):
                x = (s) / (n - 1)
                M = sigma_max
                m = sigma_min

                sigmas[s] = eval((OverSchedForge.custom))

                s += 1

        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_AYS_sd15(n, sigma_min, sigma_max, device='cpu'):
        sigmas_d = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

        xs = np.linspace(0, 1, len(sigmas_d))
        ys = np.log(sigmas_d[::-1])
        
        new_xs = np.linspace(0, 1, n)
        new_ys = np.interp(new_xs, xs, ys)
        
        interped_ys = np.exp(new_ys)[::-1].copy()

        sigmas = torch.tensor(interped_ys, device=device)

        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_AYS_sdXL(n, sigma_min, sigma_max, device='cpu'):
        sigmas_d = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]

        xs = np.linspace(0, 1, len(sigmas_d))
        ys = np.log(sigmas_d[::-1])
        
        new_xs = np.linspace(0, 1, n)
        new_ys = np.interp(new_xs, xs, ys)
        
        interped_ys = np.exp(new_ys)[::-1].copy()

        sigmas = torch.tensor(interped_ys, device=device)

        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def scale_sigmas (sigmas, sigma_min, sigma_max, device='cpu'):
        #scales sigmas to between given min/max - correction for default, ideally only a temp. fix
        #better to find and patch the used get_sigmas functions - but where is it? kdiffusion.external
        listSigmas = sigmas.tolist()
        #assume min/max at end/start
        currentMin = listSigmas[-1]
        currentMax = listSigmas[0]

        for i in range(len(listSigmas)):
            listSigmas[i] -= currentMin
            listSigmas[i] /= (currentMax - currentMin)
            listSigmas[i] *= (sigma_max - sigma_min)
            listSigmas[i] += sigma_min

        return torch.tensor(listSigmas, device=device)
        

    def setup_img2img_steps(p, steps=None):
        requested_steps = (steps or p.steps)
        steps = requested_steps
        t_enc = requested_steps - 1

        return steps, t_enc

    def calculate_sigmas (self, scheduler, steps, sigmaMin, sigmaMax): #scheduler is a parameter to enable previews (does it matter?)
        if scheduler == 'karras':
            if opts.use_old_karras_scheduler_sigmas:
                sigmaMin, sigmaMax = (0.1, 10)
            sigmas = k_diffusion.sampling.get_sigmas_karras                     (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'exponential':
            sigmas = k_diffusion.sampling.get_sigmas_exponential                (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)

        elif scheduler == 'cosine':
            sigmas = patchedKDiffusionSampler.get_sigmas_cosine                 (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)

        elif scheduler == 'phi':
            sigmas = patchedKDiffusionSampler.get_sigmas_phi                    (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'fibonacci':
            sigmas = patchedKDiffusionSampler.get_sigmas_fibonacci              (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'continuous VP':
            sigmas = k_diffusion.sampling.get_sigmas_vp                         (n=steps,                                         device=shared.device)
        elif scheduler == '4th power':
            sigmas = patchedKDiffusionSampler.get_sigmas_fourth                 (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'Align Your Steps sd15':
            sigmas = patchedKDiffusionSampler.get_sigmas_AYS_sd15               (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'Align Your Steps sdXL':
            sigmas = patchedKDiffusionSampler.get_sigmas_AYS_sdXL               (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
               
        elif scheduler == 'custom' and OverSchedForge.custom != "":
            sigmas = patchedKDiffusionSampler.get_sigmas_custom                 (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)
            sigmas = patchedKDiffusionSampler.scale_sigmas (sigmas, sigmaMin, sigmaMax)
            #evenly spaced timesteps from 999 to 0
            #uses table of log sigmas for all possible timesteps, interpolates

        return sigmas

#DiscreteSchedule.get_sigmas CompVisDenoiser

    def apply_action (action, sigmas):
        if action == None:
            return sigmas
        else:
            sigmaList = sigmas.tolist()
            steps = len(sigmaList)-1           #   -1 to ignore the zero
            sigmaMin = sigmaList[-2]
            sigmaMax = sigmaList[0]

            if action == "blend to exponential":
                K = (sigmaMin / sigmaMax)**(1/(steps-1))
                E = sigmaMax
                for x in range(steps):
                    p = x / (steps-1)
                    sigmaList[x] = sigmaList[x] + p * (E - sigmaList[x])
                    E *= K
            elif action == "blend to linear":
                E = sigmaMax**0.5
                D = (E - sigmaMin) / (steps-1)
                for x in range(steps):
                    p = x / (steps-1)
                    sigmaList[x] = sigmaList[x] + p * (E - sigmaList[x])
                    E -= D
            elif action == "threshold":
                E = sigmaMax**0.5
                D = (E - sigmaMin) / (steps-1)
                for x in range(steps):
                    sigmaList[x] = max(sigmaList[x], E)
                    E -= D

            return torch.tensor(sigmaList, device=shared.device)


    def get_sigmas(self, p, steps):

        #   restore original function ASAP, in case of problem later
        K.KDiffusionSampler.get_sigmas = OverSchedForge.get_sigmas_backup

#        hr_enabled = getattr(p, "enable_hr", False)

#        m_sigma_min, m_sigma_max = (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())
        m_sigma_min = OverSchedForge.sigmaMin
        m_sigma_max = OverSchedForge.sigmaMax

        if OverSchedForge.setup_img2img_steps_backup != None:
            #if patched, hiresAlt is enabled and this is the hires pass
            m_sigma_max *= p.denoising_strength
            steps = p.hr_second_pass_steps              


        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        if OverSchedForge.sgm == True:  #   is this really all SGM is?
            steps += 1


        sigmas = patchedKDiffusionSampler.calculate_sigmas (self, OverSchedForge.scheduler, steps, m_sigma_min, m_sigma_max)
        sigmas = patchedKDiffusionSampler.apply_action (OverSchedForge.action, sigmas)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        if OverSchedForge.sgm == True:  #   is this really all SGM is?
            sigmas = sigmas[:-1]

#       apply a scaling per sigma here

        if OverSchedForge.setup_img2img_steps_backup != None:
            modules.sd_samplers_common.setup_img2img_steps = OverSchedForge.setup_img2img_steps_backup
            OverSchedForge.setup_img2img_steps_backup = None


        return sigmas


#also need to modify sample_img2img ? seems likely, but get this fully functional first
#changing sampler mid way in img2img seems less likely to be useful, but might need to patch it anyway to tweak scheduler method


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

        s1 = torch.tensor(listSigmas[0:stepToChange+1], device='cuda:0')
        s2 = torch.tensor(listSigmas[stepToChange:len(sigmas)], device='cuda:0')

        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters: 
            extra_params_kwargs['n'] = steps

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

        #euler dy slightly different results with switch, and dpm2 (uses sigma[i-1])


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

        if extraParams.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

#maybe should control this by sampler, but seems like they have default values anyway
#add UI for these values?? overkill


        if samplerIndex == 3 or samplerIndex == 5 or samplerIndex == 6 :     #euler, heun, dpm2
            extra_params_kwargs['s_churn'] = shared.opts.s_churn
            extra_params_kwargs['s_tmin'] = shared.opts.s_tmin
            extra_params_kwargs['s_tmax'] = shared.opts.s_tmax
        elif samplerIndex == 16 or samplerIndex == 17:     #euler dy *2
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
##import pickle
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
    setup_img2img_steps_backup = None
    sgm = False
##    last_scheduler = None


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
                enabled = gr.Checkbox(value=False, label='Enabled')
                hiresAlt = gr.Checkbox(value=False, label='Use alternate method for HiRes')
                sgm = gr.Checkbox(value=False, label='SGM')
            with gr.Row(equalHeight=True):
                scheduler = gr.Dropdown(["None", "karras", "exponential", "cosine",
                                         "phi", "fibonacci", "continuous VP", "4th power",
                                         "Align Your Steps sd15", "Align Your Steps sdXL", "custom"],
                                        value="None", type='value', label='Scheduler choice', scale=1)
                action = gr.Dropdown(["None", "blend to exponential", "blend to linear", "threshold"],
                                     value="None", type="value", label="extra action", multiselect=False)
                                    
            custom = gr.Textbox(value='', label='custom function/list', lines=1.1, visible=False)
            with gr.Row():
                defMin = ToolButton (value='\U000027F3')
                sigmaMin = gr.Slider (label="sigma minimum", value=0.029168,
                                      minimum=0.001, maximum=2.0, step=0.001);
                sigmaMax = gr.Slider (label="sigma maximum", value=14.614642,
                                      minimum=2.0, maximum=30.0, step=0.001);
                defMax = ToolButton (value='\U000027F3')
            with gr.Row(equalHeight=True):
                sampler = gr.Dropdown(samplerList, value="None", type='index', label='Sampler choice', scale=1)
                step = gr.Slider(minimum=0.01, maximum=0.99, value=0.5, label='Step to change sampler')

            with gr.Accordion (open=False, label="Sigmas graph"):
                z_vis = gr.Plot(value=None, elem_id='schedride-vis', show_label=False, scale=2) 

            for i in [scheduler, action]:
                i.change(
                    fn=self.visualize,
                    inputs=[scheduler, action, sigmaMin, sigmaMax, custom],
                    outputs=[z_vis],
                    show_progress=False
                )

            def toggleCustom (scheduler):
                if scheduler == "custom":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            scheduler.change(fn=toggleCustom, inputs=[scheduler], outputs=[custom], show_progress=False)


            def defaultSigmaMin ():
                return 0.029168
            def defaultSigmaMax ():
                return 14.614642

            defMin.click(defaultSigmaMin, inputs=[], outputs=sigmaMin, show_progress=False)
            defMax.click(defaultSigmaMax, inputs=[], outputs=sigmaMax, show_progress=False)

        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("os_enabled" in d))),
            (hiresAlt, "os_hiresAlt"),
            (sgm, "os_sgm"),
            (scheduler, "os_scheduler"),
            (action, "os_action"),
            (custom, "os_custom"),
            (sigmaMin, "os_sigmaMin"),
            (sigmaMax, "os_sigmaMax"),
            (sampler, "os_sampler"),
            (step, "os_step"),
        ]

        return enabled, hiresAlt, sgm, scheduler, action, custom, sigmaMin, sigmaMax, sampler, step

    def visualize(self, scheduler, action, sigmaMin, sigmaMax, custom):
        if scheduler == "None":
           return
        if scheduler == "custom":
            if custom == "":
                return
            OverSchedForge.custom = custom

        steps = 40
        plot_color = (1, 1, 0.8, 1.0) 
        plt.rcParams.update({
            "text.color":  plot_color, 
            "axes.labelcolor":  plot_color, 
            "axes.edgecolor":  plot_color, 
            "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),  
            "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
            "ytick.labelsize": 6,
            "ytick.labelcolor": plot_color,
            "ytick.color": plot_color,
            "figure.figsize": [5, 2.5]
        })

        fig, ax = plt.subplots(layout="constrained")
        values = patchedKDiffusionSampler.calculate_sigmas (self, scheduler, steps-1, sigmaMin, sigmaMax)
        values = patchedKDiffusionSampler.apply_action (action, values)
        ax.plot(range(steps), values.tolist(), color=plot_color)

        ##  better to specify a comparison scheduler, but if custom, should also remember those settings
##        if scheduler != OverSchedForge.last_scheduler and OverSchedForge.last_scheduler != None:
##            values2 = patchedKDiffusionSampler.calculate_sigmas (OverSchedForge.last_scheduler, steps-1, sigmaMin, sigmaMax).tolist()
##            plot_color2 = (0.8, 0.4, 0.4, 1.0) 
##            ax.plot(range(steps), values2, color=plot_color2)
##            OverSchedForge.last_scheduler = scheduler

        ax.tick_params(right=False, color=plot_color)
        ax.set_xticks([i * (steps - 1) / 10 for i in range(10)][1:])
        ax.set_xticklabels([])
        ax.set_ylim([0,sigmaMax])
        ax.set_xlim([0,steps-1])
        plt.close()
        return fig   


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, hiresAlt, sgm, scheduler, action, custom, sigmaMin, sigmaMax, sampler, step = script_args
        self.enabled = enabled

        if not enabled:
            return

        OverSchedForge.sgm = sgm
        OverSchedForge.scheduler = scheduler
        OverSchedForge.action = action
        OverSchedForge.custom = custom
        OverSchedForge.sigmaMin = sigmaMin
        OverSchedForge.sigmaMax = sigmaMax
        OverSchedForge.samplerIndex = sampler
        OverSchedForge.step = step

        K.KDiffusionSampler.get_sigmas = patchedKDiffusionSampler.get_sigmas
        if hiresAlt == True and params.is_hr_pass == True:
            OverSchedForge.setup_img2img_steps_backup = modules.sd_samplers_common.setup_img2img_steps
            modules.sd_samplers_common.setup_img2img_steps = patchedKDiffusionSampler.setup_img2img_steps


        if sampler != 0:
            K.KDiffusionSampler.sample = patchedKDiffusionSampler.sample


        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(
            os_enabled = enabled,
            os_hiresAlt = hiresAlt,
            os_sgm = sgm,
            os_scheduler = scheduler,
            os_action = action,
            os_sigmaMin = sigmaMin,
            os_sigmaMax = sigmaMax,
            os_sampler = patchedKDiffusionSampler.samplers_list[sampler][0],
            ))
        if scheduler == "custom":
            params.extra_generation_params.update(dict(os_custom = custom, ))
        if sampler != 0:
            params.extra_generation_params.update(dict(os_step = step, ))

        return


