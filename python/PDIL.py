class PDIL:
    IDX_POWER: int = 0
    IDX_LIMIT: int = 1

    pdil: list[tuple[float, float]]

    def __init__(self, pdil: list[tuple[float, float]]):
        self.pdil = pdil
        self.pdil.sort()

    def getPDIL(self, power: float) -> float:

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
