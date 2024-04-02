from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True) # frozen makes it read-only/immutable, slots makes it use less memory and makes accessing faster
class TrainingPoint():
    currPosition: np.array # current position of the bot
    currAngle: float # current angle
    lidarMeasurements: list[float] # lidar measurement array at the current position
    targetPosition: np.array # desired next position along the bot's path
    endPosition: np.array # terminal position on the bot's path
