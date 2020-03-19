from gym_wmgds.envs.registration import registry, register, make, spec
import numpy as np

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym_wmgds.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym_wmgds.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym_wmgds.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym_wmgds.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym_wmgds.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym_wmgds.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym_wmgds.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym_wmgds.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCar-v0',
    entry_point='gym_wmgds.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuous-v0',
    entry_point='gym_wmgds.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym_wmgds.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='Acrobot-v1',
    entry_point='gym_wmgds.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym_wmgds.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym_wmgds.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v2',
    entry_point='gym_wmgds.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym_wmgds.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='CarRacing-v0',
    entry_point='gym_wmgds.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym_wmgds.envs.toy_text:BlackjackEnv',
)

register(
    id='KellyCoinflip-v0',
    entry_point='gym_wmgds.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,
)
register(
    id='KellyCoinflipGeneralized-v0',
    entry_point='gym_wmgds.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

register(
    id='FrozenLake-v0',
    entry_point='gym_wmgds.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym_wmgds.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='CliffWalking-v0',
    entry_point='gym_wmgds.envs.toy_text:CliffWalkingEnv',
)

register(
    id='NChain-v0',
    entry_point='gym_wmgds.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

register(
    id='Roulette-v0',
    entry_point='gym_wmgds.envs.toy_text:RouletteEnv',
    max_episode_steps=100,
)

register(
    id='Taxi-v2',
    entry_point='gym_wmgds.envs.toy_text:TaxiEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)

register(
    id='GuessingGame-v0',
    entry_point='gym_wmgds.envs.toy_text:GuessingGame',
    max_episode_steps=200,
)

register(
    id='HotterColder-v0',
    entry_point='gym_wmgds.envs.toy_text:HotterColder',
    max_episode_steps=200,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v2',
    entry_point='gym_wmgds.envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Pusher-v2',
    entry_point='gym_wmgds.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v2',
    entry_point='gym_wmgds.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v2',
    entry_point='gym_wmgds.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='gym_wmgds.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v2',
    entry_point='gym_wmgds.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v2',
    entry_point='gym_wmgds.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='gym_wmgds.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v2',
    entry_point='gym_wmgds.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='gym_wmgds.envs.mujoco:Walker2dEnv',
)

register(
    id='Ant-v2',
    entry_point='gym_wmgds.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='gym_wmgds.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v2',
    entry_point='gym_wmgds.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Robotics
# ----------------------------------------

def _merge(a, b):
    a.update(b)
    return a

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='FetchSlide{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlace{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchReach{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPush{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushPartial{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPushPartialEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushLD{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPushLDEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPushMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )    

    register(
        id='FetchPickAndPlaceMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlaceFloorMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceFloorMultiEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )

    register(
        id='FetchPickAndPlaceGrippedMultiEnv{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceGrippedMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlaceFlexMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceFlexEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlaceFlexHingedMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceFlexHingedEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideMulti{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchSlideMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )   

    for n_objects in range(2,5):
        register(
            id='FetchStackMulti{}{}-v1'.format(suffix,n_objects),
            entry_point='gym_wmgds.envs.robotics:FetchStackMultiEnv',
            kwargs=kwargs,
            #max_episode_steps=50*(n_objects-1),
            max_episode_steps=50*(np.minimum(n_objects,3)-1),
            
        )

    for n_objects in range(2,5):
        register(
            id='FetchStackBordersMulti{}{}-v1'.format(suffix,n_objects),
            entry_point='gym_wmgds.envs.robotics:FetchStackBordersMultiEnv',
            kwargs=kwargs,
            #max_episode_steps=50*(n_objects-1),
            max_episode_steps=50*(np.minimum(n_objects,3)-1),
        )

    #Multi-agent robotics

    register(
        id='FetchPickAndPlaceMaRobLong{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceMaRobLongEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlaceMaRobSeq{}-v1'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:FetchPickAndPlaceMaRobSeqEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    # Kortex
    register(
        id='KortexReach{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:KortexReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='KortexPushMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:KortexPushMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='KortexPickAndPlaceMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:KortexPickAndPlaceMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )  

    # Universal
    register(
        id='UniversalPushMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:UniversalPushMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='UniversalPickAndPlaceMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:UniversalPickAndPlaceMultiEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )  

    # Hand
    register(
        id='HandReach{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='HandManipulateBlockRotateZ{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallel{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZ{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockFull{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateBlock{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockTouchSensors{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotate{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggFull{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateEgg{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggTouchSensors{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotate{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenFull{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulatePen{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenTouchSensors{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

# Hand environments with i-object
    register(
        id='HandManipulateBlockRotateZMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallelMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockTranslateMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockFullMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateBlockMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandBlockMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotateMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggTranslateMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggFullMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateEggMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandEggMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotateMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotateXYZMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'ignore_z_target_rotation': False}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenTranslateMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'ignore'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotateMultiShifted{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiShiftedEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )
    
    register(
        id='HandManipulatePenFullMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenFullXYZMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'ignore_z_target_rotation': False}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulatePenMulti{}-v0'.format(suffix),
        entry_point='gym_wmgds.envs.robotics:HandPenMultiEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )



# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v4'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v4'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='gym_wmgds.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )


# Unit test
# ---------

register(
    id='CubeCrash-v0',
    entry_point='gym_wmgds.envs.unittest:CubeCrash',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashSparse-v0',
    entry_point='gym_wmgds.envs.unittest:CubeCrashSparse',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashScreenBecomesBlack-v0',
    entry_point='gym_wmgds.envs.unittest:CubeCrashScreenBecomesBlack',
    reward_threshold=0.9,
    )

register(
    id='MemorizeDigits-v0',
    entry_point='gym_wmgds.envs.unittest:MemorizeDigits',
    reward_threshold=20,
    )

