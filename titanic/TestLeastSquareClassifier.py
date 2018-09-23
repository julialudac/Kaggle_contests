
import csv
import numpy as np


class TestLeastSquareClassifier:
    """For this exo, no target T for the test data
    => the evaluation is not present in this piece of code
    For now, I'll estimate for the observations one by one => I can estimate just
    after the extraction"""

    def __init__(self):
        self.X = None # Test set
        self.T_hat = None # Estimations
        self.N = 0
        self.ids = None

    def extract_data_and_estimate(self, W, debug=False):
        """Extract each test observation one by one and estimate"""
        print("Extracting data")
        file = open("test.csv", "r", encoding="utf-8")

        """We initialize the data"""
        """##We detect the nb of elements in the file without ignoring any one"""
        file.readline()
        for l in file:
            self.N += 1
        file.close()
        file = open("test.csv", "r", encoding="utf-8")
        """##We can now initialize"""
        self.X = np.zeros((self.N, 7))  # In this exo, x has 7 features.
        self.T = np.zeros((self.N, 1))
        self.ids = np.zeros((self.N, 1))
        self.T_hat = np.zeros((self.N, 1))
        # print("Self.X:",self.X)

        """We read the data"""
        myreader = csv.reader(file)
        """##We skip the 1st line"""
        myreader.__next__()
        """##We will read line by line"""
        ind = 0
        alives = 0  # optional: number of alives
        for row in myreader:
            """Don't forget to collect the client's id"""
            self.ids[ind, 0] = float(row[0])
            """Be careful there are missing data => I consider dead """
            little_row = extract_useful_fields(row)
            if "" in little_row:
                self.X[ind, :] = 0 # I don't care about that since I consider dead
                self.T_hat[ind, 0] = 0
            else:
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
                """Decide"""
                xtilde = np.zeros((8, 1))
                xtilde[0, 0] = 1
                xtilde[1:8, 0] = self.X[ind, :]
                y = np.dot(np.transpose(W), xtilde)
                if y[0, 0] < 0.5:
                    self.T_hat[ind, 0] = 0
                else:
                    self.T_hat[ind, 0] = 1
                    alives += 1
            ind += 1
        print("nb alives:", alives)
        if debug:
            np.savetxt("extracted_X_test", self.X, fmt="%f")
            np.savetxt("T_hat", self.T_hat, fmt="%d")

    def output_results(self):
        """Output results in the form of string lines, one by one"""
        file = open("results.csv", "w", encoding="utf-8")
        file.write("PassengerId,Survived\n")
        for i in range(self.N):
            file.write(str(int(self.ids[i, 0])) + "," + str(int(self.T_hat[i, 0])) + "\n")


def extract_useful_fields(row):
    """Caution! Differs from the function for the training phase in Classifier.py since one field less: Survived
    Gives a subrow with the useful fields FOR AN OBSERVATION x_n.
    The useless rows indices are 0 (Passenger), 1 (Survived), 3 (Name), 8 (Ticket) and 10 (Cabin)
    """

    # We need to substract the indice n# since when we delete one element, the indices change.
    little_row = row.copy()
    little_row.pop(0)
    #little_row.pop(1-1)
    little_row.pop(3-2)
    little_row.pop(8-3)
    little_row.pop(10-4)
    return little_row
