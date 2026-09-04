# handles basic file operations
import os

class File:
    """
    A class to handle basic file operations such as reading and writing.
    """
    name = ""
    path = ""
    fullPath = ""
    fileExists = False
    verbose = True
    headerList = []
    separateHeader = False
    ignoreSomeColumns = False
    ignoreThese = []
    ignoreLength = 0

    def __init__(self, path, name):
        """
            sets our file path and tests for the existence of the file, setting an attribute and printing output.
            returns the file's existence status.
        """
        self.name = name
        self.path = path
        self.fullPath = os.path.join(self.path, self.name)
        if not os.path.isfile(self.fullPath):
            print(f"File '{self.name}' does not exist.")
        else:
            self.fileExists = True
            print(f"File '{self.name}' exists.")

    def setVerbose(self, verbose):
        """
            Set our verbose debug on or off.
        """
        self.verbose = verbose
        return True

    def doesFileExist(self):
        return self.fileExists

    def getHeader(self):
        return self.headerList

    def setColumnsToIngore(self,ignore=[]):
        """
            Expecting these as a numbered column starting with zero
        """
        self.ignoreSomeColumns = True
        self.ignoreLength = len(ignore)
        self.ignoreThese = ignore

    def readCSVIntoMatrix(self,separateHeader=True):
        """
            reads the file, by line, into a list.
            Each element is one line, w/out the new line character.
            parameters:
                separateHeader<boolean> do we split off the first row as a header or not?
        """
        counter = 0
        matrix = []
        if separateHeader == True:
            self.separatedHeader = True

        if (self.fileExists == True):
            with open(self.fullPath, 'r') as file:
                if (self.separatedHeader == True):
                    header = file.readline()
                    header = header.strip()
                    header = header.split(',')
                    if (self.ignoreSomeColumns == True):
                        header = self._stripColumns(header)
                    self.headerList = header
            
                for line in file:
                    line = line.strip()
                    line = line.split(',')
                    if (self.ignoreSomeColumns == True):
                        line = self._stripColumns(line)

                    matrix.append(line)
                    counter += 1
        else:
            raise FileNotFoundError(f"File '{self.name}' does not exist.")

        return matrix

    def _stripColumns(self,listToClean):
        offSet = 0
        for i in range(self.ignoreLength):
            thisColToStrip = self.ignoreThese[i]
            #Once you delete the first time, the resulting List is shrunk by one.
            #So we need to reduce our column list by however many times we've deleted a column.
            thisColToStrip = thisColToStrip - offSet
            offSet += 1
            del listToClean[thisColToStrip]

        return listToClean

    def read(self):
        #basic if file exists, then read the file and return.
        if self.fileExists == True:
            with open(self.fullPath, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"File '{self.name}' does not exist.")  

    def write(self, data, overWrite=False, format='',headers=''):
        """
           handles writing data to the file.
           does conversion of data types to specific formats.
           currently supports list -> csv.
        """
        if overWrite == True:
            with open(self.fullPath, 'w') as f:
                f.write("")

        match format:
            case 'csv':
                with open(self.fullPath, 'a') as f:
                    if (isinstance(data, list) == False):
                        raise ValueError("Data must be a list for list->CSV conversion. Either convert to CSV and pass into this method as a string, or pass in a list of data to be converted to CSV.  ")

                    if (self.verbose == True):
                        print(f"Writing data to file: {self.fullPath}")

                    if headers:
                        f.write(f"{headers}\n")

                    for row in data:
                        csvRow = ''.join(str(value) for value in row)
                        f.write(csvRow)
                        f.write("\n")
            case 'json':
                with open(self.fullPath, 'a') as f:
                    f.write(data)
            case _:
                with open(self.fullPath, 'a') as f:
                    f.write(data)
