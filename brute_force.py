import timeit
from typing import TypedDict


class DesignOptionParams(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int


class DesignOption:
    """
    A class to represent a design option. It is overkill for this example but
    good practice as you could imagine a class with lots of functions
    validating many aspects of a design. DETI EC5 - Digital Thread.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, params: DesignOptionParams) -> None:
        self.x1 = params["x1"]
        self.y1 = params["y1"]
        self.x2 = params["x2"]
        self.y2 = params["y2"]
        pass

    def validate(self) -> bool:
        if self.x1 == self.x2 and self.y1 == self.y2:
            return False
        if self.x1 == Grid_width-1 and self.x2 == Grid_width-1:
            return True
        return False


def brute_force_it():
    # Create the options
    design_options = []
    for i in range(0, Grid_width):
        for j in range(0, Grid_width):
            for k in range(0, Grid_width):
                for l in range(0, Grid_width):
                    params: DesignOptionParams = dict(x1=i, y1=j, x2=k, y2=l)
                    design_options.append(DesignOption(params=params))

    # Perform the checks on each one
    valid = []
    invalid = []
    for d in design_options:
        if d.validate():
            valid.append(d)
        else:
            invalid.append(d)

    return (valid, invalid)


Grid_width = 64


if __name__ == "__main__":
    # Timing the execution of the function.
    print(timeit.timeit(stmt=brute_force_it, number=50)/50) #running sccript 50 times to find average time for completion

    # Just to check it is reporting the right results.
    v, i = brute_force_it()
    print(f"Valid: {len(v)}, Invalid: {len(i)}")
