from typing import Any

class ControlRod:
    IDX_POWER: int = 0
    IDX_LIMIT: int = 1

    id: str
    distance : float
    pos: float
    pdil: list[tuple[float, float]]
    overlappedWith: Any
    overlappedRange: tuple[float, float]

    def __init__(self, id: str, distance: float, overlappedWith: Any = None, overlappedRange: tuple[float, float] = (0,0)):
        self.id = id
        self.distance = distance
        self.pos = distance
        self.overlappedWith = overlappedWith
        self.overlappedRange = overlappedRange
        self.pdil = []

    def setPosition(self, pos: float):
        self.pos = pos
    
    def setOverlappedPosition(self, pos: float):
        self.pos = pos
        if self.overlappedWith is not None: 
            maxPos = min(pos + (self.distance-self.overlappedRange[0]), self.distance)
            if(self.overlappedWith.pos > maxPos):
                self.overlappedWith.setOverlappedPosition(maxPos)

            minPos = min(pos + (self.distance-self.overlappedRange[1]), self.distance)
            if(self.overlappedWith.pos < minPos):
                self.overlappedWith.setOverlappedPosition(minPos)


    def setPDIL(self, pdil: list[tuple[float, float]]):
        self.pdil = pdil
        self.pdil.sort()

    def getPDIL(self, power: float) -> float:

        if len(self.pdil) == 0: 
            return 0.0

        currItem: tuple[float, float] = self.pdil[0]
        prevItem: tuple[float, float] = self.pdil[0]

        for item in self.pdil:
            currItem = item
            if item[self.IDX_POWER] > power:
                break
            else:
                prevItem = item

        slope = (currItem[self.IDX_LIMIT] - prevItem[self.IDX_LIMIT]) / \
            (currItem[self.IDX_POWER]-prevItem[self.IDX_POWER]+0.001)

        return slope * (power - currItem[self.IDX_POWER]) + currItem[self.IDX_LIMIT]
