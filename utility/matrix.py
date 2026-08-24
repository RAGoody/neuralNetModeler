class matrix:
    thisIsAMatrix = False
    matrix = []
    length = 0
    width = 0
    statistics = dict()

    def __init__(self,matrix):
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

        #now generate statistics about this matrix.