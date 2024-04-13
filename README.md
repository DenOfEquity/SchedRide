﻿# Scheduler Override #
### extension for Forge webui for Stable Diffusion ###
---
## Basic usage ##
*	Select a noise scheduler from the dropdown menu. When the extension is enabled, this scheduler will be used instead of the default scheduler linked to your selected Sampling method.
Settings used are saved with metadata, and restored from loading through the **PNG Info** tab.
*	Select a sampler and a step to switch to it. For example, select DPM++ SDE in the normal sampler selection, then use this to switch to Euler after 60%.
---
## Advanced / Details ##
I patch the **get_sigmas** function in **KDiffusionSampler**, then unpatch when processing has finished.
Some extra schedulers have been added:
* *polyexponential* is part of k-diffusers, normally unused.
* *phi* is a slightly tweaked version of the golden scheduler from [Extraltodeus](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler).
* *fibonacci* is a reverse fibonacci curve, scaled down by step to keep it under control a little longer. Even so, high step counts will result in a long run of very low sigmas - i.e. minimal change to results.
* *continuous VP* is also in k-diffusers, normally unused. From brief testing, seems worthwhile.
* ~~*squared* is an ultra-simple curve: step is normalised to 1.0 -> 0.0, squared, then scaled to *sigma_max*-> *sigma_min*. Again, from brief testing, seems good. Equivalent to custom function: m + (M-m)*(1-x)*(1-x)~~
* *4th power, thresholded* is a simple curve: step is normalised to 1.0 -> 0.0, taken to fourth power, maximum taken compared to linear scaled by 4/n,  then scaled to *sigma_max*-> *sigma_min*. Again, from brief testing, seems good. Equivalent to custom function: m + (M-m)*max((1-x)**4, (1-x)/(0.25*n))
* ~~*linear* is not even a curve. It's *squared* without the squaring. Sometimes has use, often does not.~~
* *custom* allows user-generated schedules using standard Python code. The custom function is evaluated at each step. The following variables are defined:
	* *m*: minimum sigma (adjustable in **Settings**, usually ~0.03)
	* *M*: maximum sigma (adjustable in **Settings**, usually ~14.6)
	* *n*: total steps
	* *s*: this step
	* *x*: step / (total steps - 1)
	* *phi*: (1 + sqrt(5)) / 2
	more may be added later


Support for Euler Dy and Euler SMEA Dy samplers requires that the relevant extension be installed.

---
## to do? ##
1. ~~option to change sampler after some number of steps. There is an extension that does this: [Seniorious by Carzit](https://github.com/Carzit/sd-webui-samplers-scheduler). I think it makes sense to include that functionality here.~~
	Now just needs improving, maybe multiple switches. Though 1 + 2 switches seems like the most that would ever be reasonable.
2. ~~import math for custom schedulers, (probably necessary)~~
3. options to set sigma limits (probably not, easy to do in Settings)


---
## License ##
Public domain. Unlicense. Free to a good home.
All terrible code is my own. I've learned from other extensions, StackOverflow, Bing Chat, banging my head against my desk, and cursing IDLE. No warranty. Check the code for yourself.

> Written with [StackEdit](https://stackedit.io/).
