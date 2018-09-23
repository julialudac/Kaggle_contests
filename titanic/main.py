
from LeastSquareClassifier import LeastSquareClassifier
from TestLeastSquareClassifier import TestLeastSquareClassifier


def main():

    print("main")
    lsc = LeastSquareClassifier()
    lsc.extract_data(True)
    lsc.train(True)
    tlsc = TestLeastSquareClassifier()
    tlsc.extract_data_and_estimate(lsc.W, True)
    tlsc.output_results()



if __name__ == "__main__":
    main()