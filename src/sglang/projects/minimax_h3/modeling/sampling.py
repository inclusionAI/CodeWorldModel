"""Deterministic Euler sampling for H3's rectified-flow ODE."""
from common.diffusion.sampler.ddim import DDIMSampler
from common.diffusion.schedule.lerp import LinearInterpolationSchedule


class EulerSampler(DDIMSampler):
    """Reuse DDIM's numerics, which equal Euler for a linear flow schedule.

    For x(t) = (1-t/T) x0 + (t/T) eps, its step to s is
    x(s) = x(t) + (s-t)/T * (eps_pred-x0_pred). There is no added noise.
    Restrict the schedule so this equivalence cannot silently stop holding.
    """

    def __init__(self, schedule, **kwargs):
        if not isinstance(schedule, LinearInterpolationSchedule):
            raise ValueError('EulerSampler requires a LinearInterpolationSchedule')
        super().__init__(schedule=schedule, **kwargs)
