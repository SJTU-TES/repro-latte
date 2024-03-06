from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler


SCHEDULER = {
    "DDIM": DDIMScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "DDPM": DDPMScheduler,
    "PNDM": PNDMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler
}


def get_scheduler(
    name: str,
    pretrained_model_path: str,
    beta_start: float=0.0001,
    beta_end: float=0.02,
    beta_schedule: str="linear",
    variance_type: str="learned_range"
):

    return SCHEDULER[name].from_pretrained(
        pretrained_model_path, 
        subfolder="scheduler",
        beta_start=beta_start, 
        beta_end=beta_end, 
        beta_schedule=beta_schedule,
        variance_type=variance_type
    )
   