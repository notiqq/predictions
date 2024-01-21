class DigitClassificationInterface:
    def predict(self, x):
        pass


class DigitCNN(DigitClassificationInterface):
    def predict(self, x):
        print("accepts data and processes into CNN format data")
        return 0

class DigitRF(DigitClassificationInterface):
    def predict(self, x):
        print("accepts data and processes into RF format data")
        return 1

class DigitRand(DigitClassificationInterface):
    def predict(self, x):
        print("accepts data and processes into RAND format data")
        return 2

class DigitClassifier:

    def __init__(self) -> None:

        self.models = {
            "rand":DigitRand,
            "rf":DigitRF,
            "cnn":DigitCNN
        }

    def predict(self, x, name="rand"):
        return self.models[name]().predict(x)


if __name__ == '__main__':
    classifier = DigitClassifier()
    classifier.predict("cnn data", name="cnn")
    classifier.predict("rf data", name="rf")