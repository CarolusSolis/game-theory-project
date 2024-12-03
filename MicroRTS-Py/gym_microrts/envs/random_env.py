import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import xml.etree.ElementTree as ET
import os
import gym_microrts
from collections import defaultdict
import random

class RandomizedMicroRTSGridModeVecEnv(MicroRTSGridModeVecEnv):
    """
    A wrapper around MicroRTSGridModeVecEnv that adds domain randomization capabilities.
    """
    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
        cycle_maps=[],
        autobuild=True,
        jvm_args=[],
        resource_randomization=True,
        resource_min=5,
        resource_max=15,
        pool_size=None,  # If None, will be set based on number of environments
        pool_refresh_frequency=100  # Refresh pool every N resets
    ):
        # Store randomization parameters
        self.resource_randomization = resource_randomization
        self.resource_min = resource_min
        self.resource_max = resource_max
        self.num_envs = num_selfplay_envs + num_bot_envs
        
        # Set pool size to at least the number of environments needed
        self.pool_size = max(pool_size or self.num_envs, self.num_envs)
        self.pool_refresh_frequency = pool_refresh_frequency
        self.reset_count = 0
        
        # Set up microrts path
        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")
        
        # Store original maps for cycling
        self.original_map_paths = map_paths
        
        # Initialize map pools
        self.map_pools = defaultdict(list)
        
        # If resource randomization is enabled, create initial map pool
        if resource_randomization:
            # Create one randomized map for each environment
            if len(map_paths) == 1:
                # If only one map provided, use it for all envs
                base_map = map_paths[0]
                map_paths = [base_map for _ in range(self.num_envs)]
            
            # Create initial pool and get first set of maps
            self._ensure_pool_filled()
            map_paths = self._get_random_maps(self.num_envs)
        
        super().__init__(
            num_selfplay_envs=num_selfplay_envs,
            num_bot_envs=num_bot_envs,
            partial_obs=partial_obs,
            max_steps=max_steps,
            render_theme=render_theme,
            frame_skip=frame_skip,
            ai2s=ai2s,
            map_paths=map_paths,
            reward_weight=reward_weight,
            cycle_maps=cycle_maps,
            autobuild=autobuild,
            jvm_args=jvm_args,
        )

    def _create_randomized_maps(self, base_map, num_maps):
        """Create new map files with randomized resource amounts."""
        randomized_paths = []
        
        for _ in range(num_maps):
            # Parse the original map
            full_path = os.path.join(self.microrts_path, base_map)
            tree = ET.parse(full_path)
            root = tree.getroot()
            
            # Randomize resources
            resources = np.random.randint(self.resource_min, self.resource_max + 1)
            
            # Update resource amounts in the XML
            for unit in root.findall(".//unit"):
                if unit.get("type") == "Resource":
                    unit.set("resources", str(resources))
            
            # Create new map file path
            new_path = full_path.replace(".xml", f"_random_{resources}.xml")
            tree.write(new_path)
            
            # Store relative path
            rel_path = os.path.relpath(new_path, self.microrts_path)
            randomized_paths.append(rel_path)
        
        return randomized_paths

    def _ensure_pool_filled(self):
        """Ensure the map pool has enough maps for each base map."""
        for base_map in self.original_map_paths:
            if len(self.map_pools[base_map]) < self.pool_size:
                new_maps = self._create_randomized_maps(base_map, self.pool_size - len(self.map_pools[base_map]))
                self.map_pools[base_map].extend(new_maps)

    def _get_random_maps(self, num_maps):
        """Get random maps from the pool, ensuring even distribution of base maps."""
        selected_maps = []
        
        if len(self.original_map_paths) == 1:
            # If only one base map, simply select random maps from its pool
            base_map = self.original_map_paths[0]
            # Allow replacement if we need more maps than pool size
            if num_maps > len(self.map_pools[base_map]):
                selected_maps = [random.choice(self.map_pools[base_map]) for _ in range(num_maps)]
            else:
                selected_maps = random.sample(self.map_pools[base_map], num_maps)
        else:
            # If multiple base maps, ensure we use all of them
            maps_per_base = num_maps // len(self.original_map_paths)
            remainder = num_maps % len(self.original_map_paths)
            
            for base_map in self.original_map_paths:
                count = maps_per_base + (1 if remainder > 0 else 0)
                remainder -= 1
                # Allow replacement if we need more maps than pool size
                if count > len(self.map_pools[base_map]):
                    selected_maps.extend([random.choice(self.map_pools[base_map]) for _ in range(count)])
                else:
                    selected_maps.extend(random.sample(self.map_pools[base_map], count))
        
        return selected_maps

    def reset(self):
        """
        Reset the environment using maps from the pre-generated pool.
        """
        if self.resource_randomization:
            self.reset_count += 1
            
            # Refresh pool if needed
            if self.reset_count % self.pool_refresh_frequency == 0:
                # Clear old map files
                for pool in self.map_pools.values():
                    for map_path in pool:
                        full_path = os.path.join(self.microrts_path, map_path)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                self.map_pools.clear()
                self._ensure_pool_filled()
            
            # Get random maps from pool
            new_maps = self._get_random_maps(self.num_envs)
            
            # Update the map paths
            self.map_paths = new_maps
            if hasattr(self, 'vec_client'):
                self.vec_client.close()
            self.start_client()
            
        return super().reset()
