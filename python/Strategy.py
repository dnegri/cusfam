from abc import *
from .Simon import *

class Strategy(ABC):    
    s:Simon

    def __init__(self, s:Simon) -> None:
        self.s = s

    @abstractmethod
    def next(self) -> bool:
        pass

    @abstractmethod
    def runStep(self, stdopt:SteadyOption, result:SimonResult) -> float:
        pass
