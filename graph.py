
from .load import loadData

class Summary:
    """
    Given parameters it will sumrise the deatials about similar startup and return it to the user.
    """
    def __init__(self, parameters):
        self.dataset = loadData(parameters)
    
    def predict(self):
        None
