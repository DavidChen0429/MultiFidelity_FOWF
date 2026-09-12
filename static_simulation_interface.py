from floris import FlorisModel
from abc import ABC, abstractmethod

import numpy as np
import moorpy as mp
import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz
import matplotlib.pyplot as plt
import warnings
import yaml

AVAILABLE_FARM_MODELS = [
    'floris'
]

AVAILABLE_MOORING_MODELS = [
    'moorpy'
]

class FarmSim(ABC):
    '''
    Base class to handle wind farm simulation. To be inherited by classes for 
    each specific simulation tool, to handle the specifics of each simulator
    '''

    def __init__(
            self,
            neutral_positions:np.ndarray=None,
            farm_model:str='FLORIS',
            moor_model:str='MoorPy',
            sim_config:str=None,
            ep_wind_spd:float=None,
            ep_wind_dir:float=None,
    ):
        
        # simulator information
        self.sim_steps = 0

        # mooring
        self.sim_config = self._read_config(sim_config)
        self.faero_x = None
        self.faero_y = None
        if moor_model.lower() in AVAILABLE_MOORING_MODELS:
            self.moor_model = moor_model.lower()
        else:
            warnings.warn(
                'mooring model not recognized, defaulting to fixed-bottom'
            )
            self.moor_model = None

        # wind farm
        self.applied_yaws = None
        assert farm_model.lower() in AVAILABLE_FARM_MODELS, \
            f'{farm_model} is not a recognized farm model'
        self.farm_model = farm_model.lower()

        # layout information
        self.neutral_positions = neutral_positions
        self.n_turbines = neutral_positions.shape[1]
        self.repositioned_layout = np.copy(neutral_positions)

        # wind information
        self.ep_wind_spd = ep_wind_spd
        self.ep_wind_dir = ep_wind_dir

    def _read_config(self, config_path):
        '''
        helper to read configs from YAML file for simulation setup
        
        :param config_path: path to YAML containing extra simulation  parameters
        '''
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                return cfg
        except FileNotFoundError:
            print(f'no config file found at {config_path}')
        except yaml.YAMLError as e:
            print(f'error parsing config file:\n{e}')

    # abstract methods to be defined in sim-specific classes
    @abstractmethod
    def make_observation(self):
        '''Observe the simulation environment state'''
        pass

    
    @abstractmethod
    def take_step(self, action):
        '''advance the simulation environment with the given action'''
        pass

    @abstractmethod
    def reset_simulator(self, reset_options):
        '''reset simulation environment'''
        pass

    @abstractmethod
    def get_Faero(self):
        '''get the farm coordinate x & y components of the aerodynamic forces'''
        pass

    @abstractmethod
    def solve_repositioning(self):
        '''convert Faero forces to repositioned layout'''
        pass

    @abstractmethod
    def get_powers(self):
        '''get the turbine-level powers across the wind farm'''
        pass

class FLORISFarmSim(FarmSim):
    '''
    interface for FLORIS-based steady state simulations
    '''
    def __init__(
            self,
            neutral_positions:np.ndarray=None,
            farm_model:str='FLORIS',
            moor_model:str='MoorPy',
            sim_config:str=None,
            sim:FlorisModel=None,
            ep_wind_spd:float=None,
            ep_wind_dir:float=None,
    ):
        '''
        :param neutral_positions: numpy array of turbine neutral layout (2, N)
        :type neutral_positions: np.ndarray
        :param farm_model: name of farm simulator model to be used
        :type farm_model: str
        :param moor_model: name of mooring system model to be used
        :type moor_model: str
        :param sim_config: path to YAML containing extra simulation  parameters
        :type sim_config: str
        :param sim: FLORIS simulator
        :type sim: FlorisModel
        '''
        super().__init__(
            neutral_positions,
            farm_model,
            moor_model,
            sim_config,
            ep_wind_spd,
            ep_wind_dir
        )

        # set FLORIS info
        assert farm_model.lower() == 'floris', \
            'farm model does not match simulation interface'
        self.sim = sim
        self.hub_heights = self.sim.core.farm.hub_heights
        self.rotor_diameters = self.sim.core.farm.rotor_diameters

        # set fixed wind parameters
        self.air_density = self.sim.core.flow_field.air_density

        # set available turbines (init as all active)
        self.available_turbines = np.ones(self.n_turbines, dtype=bool)

        # set mooring model
        self.moor = self._init_moor(moor_model)

        # warm-start state: last converged repositioned layout
        self.warm_layout = None

    def _init_moor(
                self,
                moor_model:str=None
    ):
        '''
        helper to instantiate the appropriate mooring model 
        '''
        moor_options = self.sim_config['mooring_options']
        match moor_model.lower():
            case 'moorpy':
                self.ms = MooringSystem(moor_options)

    def make_observation(self):
        '''
        observes FLORIS-based states
        
        :return observation: observation of physically meaningful state values
        '''
        # get necessary FLORIS outputs
        turbine_wind_speeds = self.sim.turbine_average_velocities[0]
        repositioned_layout = self.sim.get_turbine_layout()
        repositioned_layout_x, repositioned_layout_y = repositioned_layout
        wind_speed = self.sim.core.flow_field.wind_speeds[0]
        wind_direction = self.sim.core.flow_field.wind_directions[0]

        # combine for physical parameter stat observation
        obs = np.concatenate((
            [wind_speed],
            [wind_direction],
            turbine_wind_speeds,
            repositioned_layout_x,
            repositioned_layout_y
        ))

        return obs
    
    def take_step(
            self,
            action,
            reset=False,
            warm_start=True,
            initial_layout=None
    ):
        '''
        runs the simulation once, requiring a full vector of yaws
        
        :param action: vector of yaws for each turbine in the wind farm
        :type action: np.ndarray
        :param reset: whether this is a reset run or not (for wake-less power)
        :type reset: bool
        :param warm_start: whether to warm-start from the last converged layout
        :type warm_start: bool
        :param initial_layout: explicit initial layout guess, overrides warm-start
        :type initial_layout: np.ndarray
        '''
        
        self.sim.set(yaw_angles=[action]) # [action] -> FLORIS required shape
        self.applied_yaws = action

        if initial_layout is None and warm_start and self.warm_layout is not None:
            initial_layout = self.warm_layout

        converged, change_in_pos, i = self.solve_repositioning(
            max_iter=20, tol=1e-1, initial_layout=initial_layout
        )
        if not converged:
            warnings.warn(
                f'repositioning solve did not converge: {change_in_pos:0.6f}',
                UserWarning
            )

        repositioned_layout_x, repositioned_layout_y = self.repositioned_layout
        net_yaws = self.net_yaws

        self.sim.set(
            layout_x=repositioned_layout_x,
            layout_y=repositioned_layout_y,
            yaw_angles=net_yaws
        )

        if reset: # compute wake-less power
            self.sim.run_no_wake()
            self.greedy_no_wake_power = self.sim.get_farm_power().item()

        self.sim.run()
        self.sim_steps += 1
        self.warm_layout = self.repositioned_layout.copy()

    def reset_simulator(
            self,
            available_turbines=None,
            wind_direction=270,
            wind_speed=11.0
    ):
        '''
        reset simulator to align with values that parameterize current MDP
        
        :param available_turbines: indices of available turbines
        :param wind_direction: freestream wind direction for current episode 
        :param wind_speed: freestream speed for current episode
        '''
    
        # reset steps, layout, and yaw information
        self.sim_steps = 0
        self.warm_layout = None
        layout_x, layout_y = self.neutral_positions
        self.sim.set(layout_x=layout_x, layout_y=layout_y)
        self.applied_yaws = None
        self.net_yaws = None

        # reset wind condition
        self.ep_wind_spd = wind_speed
        self.ep_wind_dir = wind_direction
        self.sim.set(
            wind_speeds=[self.ep_wind_spd],
            wind_directions=[self.ep_wind_dir]
        )

        # reset operable turbine subset
        if available_turbines is not None:
            self.available_turbines = available_turbines

        self.sim.reset_operation()
        self.sim.set_operation_model('mixed')
        self.sim.set(disable_turbines=[~self.available_turbines])

        # run simulation with greedy policy
        self.take_step(action=np.zeros(self.n_turbines), reset=True)

        # clear stored forces
        self.faero_x, self.faero_y = None, None        

    def get_Faero(self):
        '''
        compute the aerodynamic forces acting through each turbines rotor
        '''
        # compute magnitude of aerodynamic thrust (drag) force
        ws = self.sim.turbine_average_velocities
        Cts = self.sim.get_turbine_thrust_coefficients()
        areas = 0.25 * np.pi * self.rotor_diameters**2
        density = self.air_density
        faero = 0.5 * density * areas * Cts * ws**2

        # decompose into turbine x & y components
        wd = self.sim.core.flow_field.wind_directions.flatten()
        yaws = self.sim.core.farm.yaw_angles
        alpha = np.deg2rad(270 - wd + yaws)
        faero_x, faero_y = np.cos(alpha)*faero, np.sin(alpha)*faero

        self.faero_x = faero_x.squeeze()
        self.faero_y = faero_y.squeeze()
        faero = np.column_stack((self.faero_x, self.faero_y))
        if np.isnan(faero).any():
            raise ValueError('nan values in aerodynamic forces')
        return faero

    def solve_repositioning(
            self,
            max_iter:int=25,
            tol:float=1e-3,
            initial_layout=None
    ):
        '''
        iteratively solve for the steady state repositioned layout
        
        :param max_iter: maximum number of iteration steps to conduct
        :type max_iter: int
        :param tol: convergence tolerance for the change in farm layout 
        :type tol: float
        :param initial_layout: initial guess for the layout (warm start)
        :type initial_layout: np.ndarray
        '''
        converged = False
        iteration = 0

        if initial_layout is not None:
            self.repositioned_layout = np.copy(initial_layout)
        else:
            self.repositioned_layout = np.copy(self.neutral_positions)

        # init solve with neutral positions or warm-start layout
        layout_x, layout_y = self.repositioned_layout
        self.sim.set(layout_x=layout_x, layout_y=layout_y)
        self.ms.reset()

        while not converged and iteration <= max_iter:
            # run simulation and get aero forces and repositioned layout
            self.sim.run()
            faero = self.get_Faero()
            repos_xys, ptfm_yaws = self.ms.solve(faero)

            # check convergence by change in layout
            new_pos = self.neutral_positions + repos_xys
            alpha = 0.5
            self.repositioned_layout = (
                (1 - alpha) * self.repositioned_layout + alpha * new_pos
            )
            change_in_pos = np.linalg.norm(new_pos - self.repositioned_layout)
            if change_in_pos <= tol:
                converged = True

            iteration += 1

            # reset simulator to new layout estimate
            layout_x, layout_y = self.repositioned_layout
            yaws = self.sim.core.farm.yaw_angles + ptfm_yaws
            self.net_yaws = yaws
            self.sim.set(layout_x=layout_x, layout_y=layout_y, yaw_angles=yaws)

        return converged, change_in_pos, iteration

    def get_powers(self):
        return self.sim.get_turbine_powers().squeeze()
    
    def get_wake_snapshot(
            self,
            title:str='',
            turbine_names:list=None,
            yaw_angles:np.ndarray=None,
            ax=None,
            show=False
    ):
        '''
        use FLORIS visualization tools to make a snapshot of the waked flow

        :param title: figure title
        :type title: str
        :param turbine_names: list of names to identify turbines in the farm
        :type turbine_names: list
        :param yaw_angles: turbine nacelle yaw misalignments 
        :type yaw_angles: np.ndarray
        :param ax: axis to add plot to
        :type ax: matplotlib.axes.Axes
        :param show: whether or not to show the figure
        :type show: bool
        '''
        if turbine_names is None:
            turbine_names = [f'WT-{i}' for i in range(self.n_turbines)]

        if ax is None:
            fig, ax = plt.subplots(dpi=120, figsize=(8,5), layout='constrained')
            ax.set_axis_off()

        horizontal_plane = self.sim.calculate_horizontal_plane(
            x_resolution=200,
            y_resolution=200,
            height=self.sim.core.farm.hub_heights.squeeze()[0]
        )

        flowviz.visualize_cut_plane(
            horizontal_plane,
            ax=ax,
            label_contours=True,
            title=title
        )

        layoutviz.plot_turbine_rotors(
            self.sim, ax=ax, yaw_angles=yaw_angles
        )

        layoutviz.plot_turbine_labels(
            self.sim, ax=ax, turbine_names=turbine_names
        )

        if show:
            plt.show()

'''
=============================================================================
                        Mooring system interfaces
=============================================================================
'''

class MooringSystem():
    '''
    initializes and wraps a MoorPy object, heavily following the example:
    https://github.com/NREL/MoorPy/blob/master/examples/manual_system.py
    '''
    
    def __init__(
            self,
            options:dict=None
    ):
        '''
        Docstring for __init__
        
        :param options: Parameters needed to set up MoorPy mooring system
        :type options: dict
        '''
        
        # mooring system layout options
        self.depth = options['depth'] # m
        self.anchor_angles = np.deg2rad(options['anchor_angles']) # rad
        self.anchor_2_fairlead_dist = options['anchor_2_fairlead_dist'] # m
        self.line_length = options['line_length'] # m
        self.fairlead_depth = options['fairlead_depth'] # m
        self.neutral_2_fairlead_dist = options['neutral_2_fairlead_dist'] # m

        # mooring system and platform properties
        self.type_name = options['type_name']
        self.diameter = options['diameter'] # mm
        self.material = options['material']
        self.mass = options['mass'] # kg
        self.displaced_volume = options['displaced_volume'] # m^3
        self.metacentric_radius = options['metacentric_radius'] # m
        self.waterplane_area = options['waterplane_area'] # m^2
        self.moment_arm = options['moment_arm'] # m

        # instantiate MoorPy
        ms = mp.System(depth=self.depth)

        # set MoorPy system parameters
        ms.setLineType(
            dnommm=self.diameter,
            material=self.material,
            name=self.type_name
        )

        # add free body to the system
        ms.addBody(
            mytype=0, # 0 -> free to move
            r6=np.zeros(6),# 6 DOF positional vector
            m=self.mass,
            v=self.displaced_volume,
            rM=self.metacentric_radius,
            AWP=self.waterplane_area
        )

        # loop through lines to set anchor, fairlead, and line
        for i, angle in enumerate(self.anchor_angles):
            r_anchor = [
                self.anchor_2_fairlead_dist*np.cos(angle),
                self.anchor_2_fairlead_dist*np.sin(angle),
                -self.depth
            ]

            r_fairlead = [
                self.neutral_2_fairlead_dist*np.cos(angle),
                self.neutral_2_fairlead_dist*np.sin(angle),
                self.fairlead_depth
            ]

            # anchor point for current line
            ms.addPoint(
                mytype=1, # 1 -> fixed point
                r=r_anchor
            )

            # fairlead point for current line
            ms.addPoint(
                mytype=1,
                r=r_fairlead
            )

            # attach fairlead point to the body
            ms.bodyList[0].attachPoint(2*i+2, r_fairlead)

            # connect the line
            ms.addLine(
                lUnstr=self.line_length,
                lineType=self.type_name,
                nSegs=20, # NOTE: may want to play with this for convergence
                pointA=2*i+1,
                pointB=2*i+2
            )

        # set mooring system
        self.ms = ms

        # init mooring system and test solve
        self.ms.initialize()
        try:
            self.ms.solveEquilibrium3()
        except:
            raise RuntimeError('issue initializing the MoorPy mooring solver')
        
    def solve(
            self,
            forces:np.ndarray
    ):
        '''
        Docstring for solve
        
        :param forces: aerodynamic (x & y) forces acting on each turbines (2, N)
        :type forces: np.ndarray
        :return positions: turbine movements relative to their neutral positions
        :type positions: np.ndarray
        '''        
        repos_xy = []
        ptfm_yaws = []
        for fx, fy in forces:
            self.reset()
            if np.linalg.norm((fx, fy)) < 1e1: # dont resolve for small force
                repos_xy.append(np.zeros(2))
                ptfm_yaws.append(0)
            else:
                self.ms.bodyList[0].f6Ext = np.array([
                    fx, fy, 0, 0, fx*self.moment_arm, 0
                ]) 
                self.ms.solveEquilibrium3()
                repos_xy.append(self.ms.bodyList[0].r6[[0, 1]])
                ptfm_yaws.append(np.rad2deg(self.ms.bodyList[0].r6[3]))

        # NOTE: Currently do not solve for z-direction movement and platform
        #       yaw contribution assumed to be negligible (returned as 0). Could
        #       add platform yaws by adjusting angle to be relative to wind dir.
        return np.array(repos_xy).T, np.array(0)

    def reset(self):
        self.ms.bodyList[0].r6[:] = 0.0
        self.ms.bodyList[0].f6Ext = np.zeros(6)