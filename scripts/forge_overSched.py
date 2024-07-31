import gradio as gr
import inspect, os
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
import torch, math, random
import torchvision.transforms.functional as TF
from PIL import Image
from modules.processing import get_fixed_seed

import numpy
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.ui_components import ToolButton                                                     

from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup
import colourPresets


class patchedKDiffusionSampler(modules.sd_samplers_common.Sampler):
    samplers_list = [
        ('No change',   None,                                           {}                                                                              ),
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
    try:
        import importlib
        EulerDy = importlib.import_module("extensions.Euler-Smea-Dyn-Sampler.smea_sampling")

        samplers_list.extend([ ("Euler Dy",         EulerDy.sample_euler_dy,            {} ), ])
        samplers_list.extend([ ("Euler SMEA Dy",    EulerDy.sample_euler_smea_dy,       {} ), ])
        samplers_list.extend([ ("Euler Negative",   EulerDy.sample_euler_negative,      {} ), ])
        samplers_list.extend([ ("Euler Negative Dy",EulerDy.sample_euler_dy_negative,   {} ), ])

    except:
        print ("Scheduler Override: Smea sampling extension not found.")

        #also get function name from all_samplers?
        
        #20 next

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
        #   some safety checks ?
        if 'import' in OverSchedForge.custom:
            sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        elif 'eval' in OverSchedForge.custom:
            sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        elif 'os' in OverSchedForge.custom:
            sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        elif 'scripts' in OverSchedForge.custom:
            sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)

        elif OverSchedForge.custom[0] == '[' and OverSchedForge.custom[-1] == ']':
#            sigmasList = eval(OverSchedForge.custom)
            sigmasList = [float(x) for x in OverSchedForge.custom.strip('[]').split(',')]

            xs = numpy.linspace(0, 1, len(sigmasList))
            ys = numpy.log(sigmasList[::-1])
            
            new_xs = numpy.linspace(0, 1, n)
            new_ys = numpy.interp(new_xs, xs, ys)
            
            interpolated_ys = numpy.exp(new_ys)[::-1].copy()
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

        xs = numpy.linspace(0, 1, len(sigmas_d))
        ys = numpy.log(sigmas_d[::-1])
        
        new_xs = numpy.linspace(0, 1, n)
        new_ys = numpy.interp(new_xs, xs, ys)
        
        interped_ys = numpy.exp(new_ys)[::-1].copy()

        sigmas = torch.tensor(interped_ys, device=device)

        return torch.cat([sigmas, sigmas.new_zeros([1])])


    def get_sigmas_AYS_sdXL(n, sigma_min, sigma_max, device='cpu'):
        sigmas_d = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]

        xs = numpy.linspace(0, 1, len(sigmas_d))
        ys = numpy.log(sigmas_d[::-1])
        
        new_xs = numpy.linspace(0, 1, n)
        new_ys = numpy.interp(new_xs, xs, ys)
        
        interped_ys = numpy.exp(new_ys)[::-1].copy()

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
            sigmas = k_diffusion.sampling.get_sigmas_karras             (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'exponential':
            sigmas = k_diffusion.sampling.get_sigmas_exponential        (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'cosine':
            sigmas = patchedKDiffusionSampler.get_sigmas_cosine         (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'phi':
            sigmas = patchedKDiffusionSampler.get_sigmas_phi            (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'fibonacci':
            sigmas = patchedKDiffusionSampler.get_sigmas_fibonacci      (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'continuous VP':
            sigmas = k_diffusion.sampling.get_sigmas_vp                 (n=steps,                                         device=shared.device)
        elif scheduler == '4th power':
            sigmas = patchedKDiffusionSampler.get_sigmas_fourth         (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
        elif scheduler == 'Align Your Steps':
            if shared.sd_model.is_sdxl == True:
                sigmas = patchedKDiffusionSampler.get_sigmas_AYS_sdXL   (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
            elif shared.sd_model.is_sd1 == True:
                sigmas = patchedKDiffusionSampler.get_sigmas_AYS_sd15   (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
            else:   #   fall back to default
                sigmas = self.model_wrap.get_sigmas(steps)
                sigmas = patchedKDiffusionSampler.scale_sigmas (sigmas, sigmaMin, sigmaMax)
                
#            sigmas = patchedKDiffusionSampler.scale_sigmas (sigmas, sigmaMin, sigmaMax)
        elif scheduler == 'custom' and OverSchedForge.custom != "":
            sigmas = patchedKDiffusionSampler.get_sigmas_custom         (n=steps, sigma_min=sigmaMin, sigma_max=sigmaMax, device=shared.device)
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
        #   restore original functions ASAP, in case of problem later
        K.KDiffusionSampler.get_sigmas = OverSchedForge.get_sigmas_backup

        if OverSchedForge.hiresAlt != "default":
            if p.hr_second_pass_steps > 0:
                steps = p.hr_second_pass_steps              
            modules.sd_samplers_common.setup_img2img_steps = OverSchedForge.setup_img2img_steps_backup

        m_sigma_min = OverSchedForge.sigmaMin
        m_sigma_max = OverSchedForge.sigmaMax

        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        if OverSchedForge.sgm == True:
            steps += 1

        if OverSchedForge.hiresAlt == "scale max sigma":
            m_sigma_max *= p.denoising_strength
        elif OverSchedForge.hiresAlt == "linear":
            m_sigma_max *= p.denoising_strength
            m_sigma_min *= p.denoising_strength
            sigmas = torch.linspace(m_sigma_max, m_sigma_min, steps, device=shared.device)
            return torch.cat([sigmas, sigmas.new_zeros([1])])
        #other methods?

        if steps == 1:
            sigmas = torch.tensor([m_sigma_max**0.5, 0.0], device=shared.device)
        else:
            sigmas = patchedKDiffusionSampler.calculate_sigmas (self, OverSchedForge.scheduler, steps, m_sigma_min, m_sigma_max)
            sigmas = patchedKDiffusionSampler.apply_action (OverSchedForge.action, sigmas)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        if OverSchedForge.sgm == True:
            sigmas = sigmas[:-1]

        return sigmas

    # via extraltodeus
    # found at https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
    # which was ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
    def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


    def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
        noise = torch.zeros(shape)
        frequency = 1
        amplitude = 1
        minDim = min(shape[0], shape[1])
        
        for _ in range(octaves):
            noise += amplitude * patchedKDiffusionSampler.rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
            frequency *= 2
            if shape[0] % frequency != 0:
                break
            if shape[1] % frequency != 0:
                break
            amplitude *= persistence
        noise = torch.remainder(torch.abs(noise)*1000000,17)/17
        return noise
    

    def create_noisy_latents_perlin(seed, width, height, batch_size, detail_level):
        noise = torch.zeros((batch_size, 4, height, width), dtype=torch.float32, device="cpu").cpu()

        if "(1 octave)" in OverSchedForge.noise:
            octaves = 1
        elif "(2 octaves)" in OverSchedForge.noise:
            octaves = 2
        elif "(4 octaves)" in OverSchedForge.noise:
            octaves = 4
        elif "(max octaves)" in OverSchedForge.noise:
            octaves = 99
        else:
            octaves = 1

        for i in range(batch_size):
            torch.manual_seed(seed + i)

            for j in range(4):
                noise_values = patchedKDiffusionSampler.rand_perlin_2d_octaves((height, width), (1,1), octaves, 0.5)
                noise_values -= 0.5 * noise_values.max()
                noise_values *= 2
                result = (1+detail_level/10)*torch.erfinv(noise_values) * (2 ** 0.5)
                result = torch.clamp(result,-4,4)

                noise[i, j, :, :] = result
        return noise


    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        #   restore original function immediately, in case of failure later means the main extension can't remove it
        K.KDiffusionSampler.sample = OverSchedForge.sample_backup

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas = self.model_wrap.sigmas.to(x.device)

        steps = steps or p.steps

        sigmas = K.KDiffusionSampler.get_sigmas(self, p, steps).to(x.device)

        w = x.size(3)
        h = x.size(2)
        n = x.size(0)
        seed = p.seed

        detail = 0.0  #   range -1 to 1

        #   can modify initial noise here? yep
        if "Perlin" in OverSchedForge.noise:
            x = patchedKDiffusionSampler.create_noisy_latents_perlin (seed, w, h, n, detail).to('cuda')
          
        if OverSchedForge.centreNoise:
            for b in range(len(x)):
                for c in range(4):
                    x[b][c] -= x[b][c].mean()

        if OverSchedForge.lowDNoise:
            for b in range(len(x)): #3,5,9
                blur2 = TF.gaussian_blur(x[b], 3)
                blur4 = TF.gaussian_blur(x[b], 5)
                blur8 = TF.gaussian_blur(x[b], 9)
                x[b] = (0.0375 * blur8) + (0.0375 * blur4) + (0.075 * blur2) + (0.985 * x[b])

        #   sharpen the initial noise, using trial derived values
        if OverSchedForge.sharpNoise:
            minDim = 1 + 2 * (min(w, h) // 2)
            for b in range(len(x)):
                blurred = TF.gaussian_blur(x[b], minDim)
                x[b] = 1.04*x[b] - 0.04*blurred




#   clamp noise
#   set all latent channels to same value

        #   colour the initial noise
        if OverSchedForge.noiseStrength != 0.0:
            nr = ((OverSchedForge.initialNoiseR ** 0.5) * 2) - 1.0
            ng = ((OverSchedForge.initialNoiseG ** 0.5) * 2) - 1.0
            nb = ((OverSchedForge.initialNoiseB ** 0.5) * 2) - 1.0

            imageR = torch.tensor(numpy.full((8,8), (nr), dtype=numpy.float32))
            imageG = torch.tensor(numpy.full((8,8), (ng), dtype=numpy.float32))
            imageB = torch.tensor(numpy.full((8,8), (nb), dtype=numpy.float32))
            image = torch.stack((imageR, imageG, imageB), dim=0)
            image = image.unsqueeze(0)

            latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), p.sd_model)

            if shared.sd_model.is_sd1 == True:
                latent *= 3.5
            latent = latent.repeat (x.size(0), 1, h, w)

            #   effect seems reduced with sdxl, so here's a hack
            strength = OverSchedForge.noiseStrength
            if shared.sd_model.is_sdxl == True:
                strength *= 2.0 ** 0.5

            #   method 0: mean stays approximately the sames
            torch.lerp (x, latent, strength, out=x)
            #   method 1: mean moves toward colour
            #x += latent * OverSchedForge.noiseStrength
                
            del imageR, imageG, imageB, image, latent

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
        samplerIndex = OverSchedForge.samplerIndex

        if samplerIndex == 0:
            s1 = sigmas.to('cuda')
        else:
            stepToChange = int(OverSchedForge.step * len(sigmas))
            s1 = torch.tensor(listSigmas[0:stepToChange+1], device='cuda:0')
            s2 = torch.tensor(listSigmas[stepToChange:len(sigmas)], device='cuda:0')

        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters: 
            extra_params_kwargs['n'] = steps

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = OverSchedForge.sigmaMin  #self.model_wrap.sigmas[-1].item()
            extra_params_kwargs['sigma_max'] = OverSchedForge.sigmaMax  #self.model_wrap.sigmas[0].item()

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

        if samplerIndex != 0:
            self.func = patchedKDiffusionSampler.samplers_list[samplerIndex][1]
            extraParams = patchedKDiffusionSampler.samplers_list[samplerIndex][2]

            parameters = inspect.signature(self.func).parameters

            if 'n' in parameters:
                extra_params_kwargs['n'] = steps

            if 'sigma_min' in parameters:
                extra_params_kwargs['sigma_min'] = OverSchedForge.sigmaMin  #self.model_wrap.sigmas[-1].item()
                extra_params_kwargs['sigma_max'] = OverSchedForge.sigmaMax  #self.model_wrap.sigmas[0].item()

            if 'sigmas' in parameters:
                extra_params_kwargs['sigmas'] = s2
            else:
                extra_params_kwargs.pop('sigmas', None)

            if extraParams.get('brownian_noise', False):
                noise_sampler = self.create_noise_sampler(x, sigmas, p)     #   should this use samples instead of x?
                extra_params_kwargs['noise_sampler'] = noise_sampler
                extra_params_kwargs['s_noise'] = 1.0
                extra_params_kwargs['eta'] = 1.0
            else:
                extra_params_kwargs.pop('eta', None)
                extra_params_kwargs.pop('s_noise', None)
                extra_params_kwargs.pop('noise_sampler', None)

            if extraParams.get('solver_type', None) == 'heun':
                extra_params_kwargs['solver_type'] = 'heun'

            if samplerIndex == 3 or samplerIndex == 5 or samplerIndex == 6 :     #euler, heun, dpm2
                extra_params_kwargs['s_churn'] = shared.opts.s_churn
                extra_params_kwargs['s_tmin'] = shared.opts.s_tmin
                extra_params_kwargs['s_tmax'] = shared.opts.s_tmax
            elif samplerIndex >= 16 and samplerIndex <= 19:     #euler dy *4
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
    centreNoise = False
    sharpNoise = False
    lowDNoise = False
##    last_scheduler = None

    from colourPresets import presetList

    def __init__(self):
        OverSchedForge.get_sigmas_backup = K.KDiffusionSampler.get_sigmas
        OverSchedForge.sample_backup = K.KDiffusionSampler.sample
        OverSchedForge.setup_img2img_steps_backup = modules.sd_samplers_common.setup_img2img_steps

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
                sgm = gr.Checkbox(value=False, label='SGM')
                hiresAlt = gr.Dropdown(["default", "scale max sigma", "linear"], value="default", type="value", label='HiRes method')
            with gr.Row(equalHeight=True):
                scheduler = gr.Dropdown(["None", "simple", "karras", "exponential", "cosine",
                                         "phi", "fibonacci", "continuous VP", "4th power",
                                         "Align Your Steps", "custom"],
                                        value="None", type='value', label='Scheduler choice', scale=1)
                action = gr.Dropdown(["None", "blend to exponential", "blend to linear", "threshold"],
                                     value="None", type="value", label="extra action")
                                    
            custom = gr.Textbox(value='', label='custom function/list', lines=1.1, visible=False)
            with gr.Row():
                defMin = ToolButton (value='\U000027F3')
                sigmaMin = gr.Slider (label="sigma minimum", value=0.029168,
                                      minimum=0.001, maximum=2.0, step=0.001);
                sigmaMax = gr.Slider (label="sigma maximum", value=14.614642,
                                      minimum=2.0, maximum=30.0, step=0.001);
                defMax = ToolButton (value='\U000027F3')

            with gr.Accordion (open=False, label="Sigmas graph"):
                z_vis = gr.Plot(value=None, elem_id='schedride-vis', show_label=False, scale=2) 

            with gr.Row(equalHeight=True):
                sampler = gr.Dropdown(samplerList, value="No change", type='index', label='Sampler choice', scale=1)
                step = gr.Slider(minimum=0.01, maximum=0.99, value=0.5, label='Step to change sampler')

            with gr.Accordion (open=False, label="Initial noise"):
                with gr.Row(equalHeight=True):
                    delPreset = ToolButton(value="-", variant='secondary', tooltip='remove preset')
                    preset = gr.Dropdown([i[0] for i in self.presetList], value="(None)", type='value', label='Colour presets', allow_custom_value=True)
                    addPreset = ToolButton(value="+", variant='secondary', tooltip='add preset')
                    savePreset = ToolButton(value="save", variant='secondary', tooltip='save presets')
                    noise = gr.Dropdown(["default", "Perlin (1 octave)", "Perlin (2 octaves)", "Perlin (4 octaves)", "Perlin (max octaves)"], type="value", value="default", label='noise type', scale=0)
                    centreNoise = ToolButton(value="c", variant='secondary', tooltip='Centre initial noise')
                    lowDNoise = ToolButton(value="d", variant='secondary', tooltip='low discrepancy noise')
                    sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gr.Row():
                    initialNoiseR = gr.Slider(minimum=0, maximum=1.0, value=0.0, label='red')
                    initialNoiseG = gr.Slider(minimum=0, maximum=1.0, value=0.0, label='green')
                    initialNoiseB = gr.Slider(minimum=0, maximum=1.0, value=0.0, label='blue')
                    noiseStrength = gr.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                
            for i in [scheduler, action, custom, sigmaMin, sigmaMax]:
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

            def updateColours (preset, nR, nG, nB, nS):
                for i in range(len(self.presetList)):
                    p = self.presetList[i]
                    if p[0] == preset:
                        return p[1], p[2], p[3], p[4]
                return nR, nG, nB, nS

            scheduler.change(fn=toggleCustom, inputs=[scheduler], outputs=[custom], show_progress=False)
            preset.change(fn=updateColours, inputs=[preset, initialNoiseR, initialNoiseG, initialNoiseB, noiseStrength], outputs=[initialNoiseR, initialNoiseG, initialNoiseB, noiseStrength], show_progress=False)

            def defaultSigmaMin ():
                return 0.029168
            def defaultSigmaMax ():
                return 14.614642
            defMin.click(defaultSigmaMin, inputs=[], outputs=sigmaMin, show_progress=False)
            defMax.click(defaultSigmaMax, inputs=[], outputs=sigmaMax, show_progress=False)

            def toggleCentre ():
                OverSchedForge.centreNoise ^= True
                return gr.Button.update(value=['c', 'C'][OverSchedForge.centreNoise],
                                        variant=['secondary', 'primary'][OverSchedForge.centreNoise])
            def togglelowD ():
                OverSchedForge.lowDNoise ^= True
                return gr.Button.update(value=['d', 'D'][OverSchedForge.lowDNoise],
                                        variant=['secondary', 'primary'][OverSchedForge.lowDNoise])
            def toggleSharp ():
                OverSchedForge.sharpNoise ^= True
                return gr.Button.update(value=['s', 'S'][OverSchedForge.sharpNoise],
                                        variant=['secondary', 'primary'][OverSchedForge.sharpNoise])

            def addColourPreset (name, r, g, b, s):
                namelist = [i[0] for i in self.presetList]
                if name not in namelist:
                    self.presetList.append((name, r, g, b, s))
                    self.presetList = sorted(self.presetList)
                return gr.Dropdown.update(choices=[i[0] for i in self.presetList])
            def delColourPreset (name):
                for i in range(len(self.presetList)):
                    if name != "(None)" and self.presetList[i][0] == name:
                        del (self.presetList[i])
                        break
                return gr.Dropdown.update(choices=[i[0] for i in self.presetList])
            def saveColourPreset ():
                #sort alphabetically? or button for that
                file = os.path.abspath(colourPresets.__file__)
                text = "presetList = [\n\t" + ',\n\t'.join(map(str, self.presetList)) +"\n]"
                with open(file, 'w') as f:
                    f.write(text)

            centreNoise.click(toggleCentre, inputs=[], outputs=centreNoise)
            lowDNoise.click(togglelowD, inputs=[], outputs=lowDNoise)
            sharpNoise.click(toggleSharp, inputs=[], outputs=sharpNoise)
            addPreset.click(addColourPreset, inputs=[preset, initialNoiseR, initialNoiseG, initialNoiseB, noiseStrength], outputs=preset)
            delPreset.click(delColourPreset, inputs=preset, outputs=preset)
            savePreset.click(saveColourPreset, inputs=[], outputs=[])

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
            (noiseStrength, "os_noiseStr"),
            (initialNoiseR, "os_nR"),
            (initialNoiseG, "os_nG"),
            (initialNoiseB, "os_nB"),
            (noise, "os_noise"),
        ]

        return enabled, hiresAlt, sgm, scheduler, action, custom, sigmaMin, sigmaMax, initialNoiseR, initialNoiseG, initialNoiseB, noiseStrength, sampler, step, noise

    def visualize(self, scheduler, action, sigmaMin, sigmaMax, custom):
        if scheduler == "None" or scheduler == "simple":
           return
        if scheduler == "custom":
            if custom == "":
                return
            try:
                m, M, x, pi, phi, n, s = 1, 1, 1, 1, 1, 1, 1
                dummy = eval(custom.strip())
                OverSchedForge.custom = custom.strip()
            except:
                return

        steps = 35          # shared.state.sampling_steps not updated until Generate
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
        if steps == 1:
            values = [sigmaMax**0.5, 0.0]
        else:
            values = patchedKDiffusionSampler.calculate_sigmas (self, scheduler, steps-1, sigmaMin, sigmaMax)
            values = patchedKDiffusionSampler.apply_action (action, values).tolist()

        ax.plot(range(steps), values, color=plot_color)

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


    def process(self, params, *script_args, **kwargs):
        enabled, hiresAlt, sgm, scheduler, action, custom, sigmaMin, sigmaMax, initialNoiseR, initialNoiseG, initialNoiseB, noiseStrength, sampler, step, noise = script_args

        if not enabled:
            return

        OverSchedForge.hiresAlt = "default"
        OverSchedForge.sgm = sgm
        OverSchedForge.scheduler = scheduler
        OverSchedForge.action = action
        OverSchedForge.custom = custom.strip()
        OverSchedForge.sigmaMin = sigmaMin
        OverSchedForge.sigmaMax = sigmaMax
        OverSchedForge.initialNoiseR = initialNoiseR
        OverSchedForge.initialNoiseG = initialNoiseG
        OverSchedForge.initialNoiseB = initialNoiseB
        OverSchedForge.noiseStrength = noiseStrength
        OverSchedForge.samplerIndex = sampler
        OverSchedForge.step = step
        OverSchedForge.noise = noise

        params.extra_generation_params.update({
            "os_enabled"        :   enabled,
            "os_hiresAlt"       :   hiresAlt,
            "os_sgm"            :   sgm,
            "os_scheduler"      :   scheduler,
            "os_action"         :   action,
            "os_sigmaMin"       :   sigmaMin,
            "os_sigmaMax"       :   sigmaMax,
            "os_sampler"        :   patchedKDiffusionSampler.samplers_list[sampler][0],
            "os_noise"          :   OverSchedForge.noise,
            "os_centreNoise"    :   OverSchedForge.centreNoise,
            "os_lowDNoise"      :   OverSchedForge.lowDNoise,
            "os_sharpNoise"     :   OverSchedForge.sharpNoise,
            "os_noiseStr"       :   noiseStrength,
            })
        if scheduler == "custom":
            params.extra_generation_params.update(dict(os_custom = custom, ))
        if sampler != 0:
            params.extra_generation_params.update(dict(os_step = step, ))
        if noiseStrength != 0:
            params.extra_generation_params.update(dict(os_nR = initialNoiseR, os_nG = initialNoiseG, os_nB = initialNoiseB, ))
            


        return

    def before_process (self, params, *args):
        enabled = args[0]
        if enabled and params.seed == -1:
            params.seed = get_fixed_seed(params.seed)
        
    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.
        enabled = script_args[0]
        hiresAlt = script_args[1]
        if enabled:
            if OverSchedForge.scheduler != "None":
                K.KDiffusionSampler.get_sigmas = patchedKDiffusionSampler.get_sigmas
            if hiresAlt != "default" and params.is_hr_pass == True:
                K.KDiffusionSampler.get_sigmas = patchedKDiffusionSampler.get_sigmas
                OverSchedForge.hiresAlt = hiresAlt
                modules.sd_samplers_common.setup_img2img_steps = patchedKDiffusionSampler.setup_img2img_steps

            K.KDiffusionSampler.sample = patchedKDiffusionSampler.sample

        return


