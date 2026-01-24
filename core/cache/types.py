class NeverRunType:
    def __eq__(self, other):
        return isinstance(other, NeverRunType)

    def __hash__(self):
        return hash("NeverRunType")

    def __repr__(self):
        return "NeverRun"

    def __str__(self):
        return "NeverRun"


NeverRun = NeverRunType()
