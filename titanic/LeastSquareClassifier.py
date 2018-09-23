
import numpy as np
import csv

from Classifier import Classifier


class LeastSquareClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.N = 0
        self.W = None # also contains the bias
        self.ids = None  # necessary since I ignore some observations
    
    def extract_data(self, debug=False):
        """We are in the classifier => extract training data of course"""
        print("Extracting data")
        file = open("train.csv", "r", encoding="utf-8")

        """We initialize the data"""
        """##We detect the nb of elements in the file THAT ARE NOT IGNORED"""
        myreader = csv.reader(file)
        myreader.__next__()  # We don't want to count the header
        for row in myreader:
            row = extract_useful_fields(row)
            #print("useful things:", row)
            if not "" in row:
                self.N +=1
        file.close()
        file = open("train.csv", "r", encoding="utf-8")
        """##We can now initialize"""
        self.X = np.zeros((self.N, 7)) # In this exo, x has 7 features.
        self.T = np.zeros((self.N, 1))
        self.ids = np.zeros((self.N, 1))
        #print("Self.X:",self.X)

        """We read the data"""
        myreader = csv.reader(file)
        """##We skip the 1st line"""
        myreader.__next__()
        """##We will read line by line"""
        ind = 0
        for row in myreader:
            """Be careful there are missing data => ignore row if missing info for a useful field """
            #print(row)
            little_row = extract_useful_fields(row)
            if "" in little_row:
                continue
            """Don't forget to collect the client's id"""
            self.ids[ind, 0] = float(row[0])
            """Load T[n]"""
            self.T[ind, 0] = float(row[1])
            """Load X[n]. Now we only keep the useful fields to fill x_n"""
            self.X[ind, 0] = float(little_row[0])
            if little_row[1] == "male":
                self.X[ind, 1] = 0
            else:
                self.X[ind, 1] = 1
            self.X[ind, 2] = float(little_row[2])
            self.X[ind, 3] = float(little_row[3])
            self.X[ind, 4] = float(little_row[4])
            self.X[ind, 5] = float(little_row[5])
            if little_row[6] == "C":
                self.X[ind, 6] = 0
            elif little_row[6] == "Q":
                self.X[ind, 6] = 1
            else:
                self.X[ind, 6] = 2
            ind += 1

        if debug:
            np.savetxt("extracted_X", self.X, fmt="%f")
            np.savetxt("extracted_T", self.T, fmt="%d")

    def train(self, debug=False):
        """Estimating W tilde"""
        Xtilde = np.zeros((self.N, 8))
        onesMatr = np.ones((1, self.N))
        Xtilde[:, 0] = np.ones(self.N) # Xtilde[:, 0] = np.transpose(onesMatr)
        Xtilde[:, 1:8] = self.X
        Wtilde = np.dot(np.linalg.pinv(Xtilde), self.T)
        self.W = Wtilde
        if debug:
            np.savetxt("estimates", self.W, fmt="%f")


def extract_useful_fields(row):
    """Gives a subrow with the useful fields FOR AN OBSERVATION x_n.
    The useless rows indices are 0 (Passenger), 1 (Survived), 3 (Name), 8 (Ticket) and 10 (Cabin)
    """

    # We need to substract the indice n# since when we delete one element, the indices change.
    little_row = row.copy()
    little_row.pop(0)
    little_row.pop(1-1)
    little_row.pop(3-2)
    little_row.pop(8-3)
    little_row.pop(10-4)
    return little_row