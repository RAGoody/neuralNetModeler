class Matrix:
    thisIsAMatrix = False
    length = 0
    width = 0

    def __init__(self,matrix):
        self.matrix = []
        self.statistics = dict()
        if (type(matrix) == list and len(matrix) > 0 and len(matrix[0]) > 0):
            self.matrix = matrix
            self.length = len(matrix)
            self.width = len(matrix[0])
            self.thisIsAMatrix = True
        else:
            self.thisIsAMatrix = False

    def isMatrix(self):
        return self.thisIsAMatrix

    def getMatrix(self):
        return self.matrix

    def getMatrixPoint(self,x,y):
        if (x < self.length and y < self.width):
            return self.matrix[x][y]
        else:
            return False

    def getRowAsList(self,row):
        if (self.isMatrix() == True):
            thisRow = []
            for i in range(self.width):
                thisRow.append(self.matrix[row][i])

            return thisRow

    def getLength(self):
        return self.length

    def getWidth(self):
        return self.width

    def generateStatistics(self):
        if (self.isMatrix == False):
            return False

        #TODO: now generate statistics about this matrix.

    def normalize(self):
        """
            Adjusts the matrix data so that it is normalized between 0 and 1 using min-max.
        """
        for column in range(self.width):
            values = [float(row[column]) for row in self.matrix]
            minVal = min(values)
            maxVal = max(values)

            if (minVal == maxVal):
                continue
            for row in self.matrix:
                row[column] = (float(row[column]) - minVal) / (maxVal - minVal)