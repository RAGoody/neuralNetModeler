# handles CLI variable reading
# initializes a cliParameters object
import argparse

class CLI:
    """
    A class to handle command-line interface (CLI) parameters.
    """
    parameters = {}

    def __init__(self,parameters):
        self.parser = argparse.ArgumentParser(description='CLI Utility')
        self.parser.add_argument("pairs", nargs="*", help="Parameters in key=value pairs. Files expected in the appropriate data/input or data/output directories.")
        args = self.parser.parse_args()

        for pair in args.pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)  # split only on the first '='
                self.parameters[key] = self._typeParameter(value)
                
            else:
                print(f"Warning: Skipping invalid parameter format: {pair}")

    def getParameter(self, parameter):
        return self.parameters.get(parameter, None)

    def getParameters(self):
        return self.parameters

    def _typeParameter(self,value):
        useTemp = False
        match value.lower(): #Are ya a boolean disguised as a silly string!?
            case 'true': 
                return True #now go away or I shall taunt you-ah ah second time-ah
            case 'false':
                return False #now go away or I shall taunt you-ah ah second time-ah
        
        try: #Are you an integer??
            temp = int(value) 
            useTemp = True #Yes? a hopeful true, to be overridden in the except on a bad attempt.
        except: 
            useTemp = False #No! Not really an integer.

        if (useTemp == True):
            return temp #off the bridge you go!
            
        try: #Arrrrre you a floater??
            temp = float(value)
            useTemp = True #Yes? a hopeful true, to be overridden in the except on a bad attempt.
        except:
            useTemp = False #No! Not really a floater.

        if (useTemp == True):
            return temp #off the bridge you go!

        #Failing the type conversions to boolean, int, & float, then just return the parameter. It's probably a string.
        return value
