import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PoppyTorso-v0',
    entry_point='gym_PoppyTorso.envs:PoppyTorsoEnv',
)
