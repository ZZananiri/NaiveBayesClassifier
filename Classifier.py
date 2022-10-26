from functools import reduce

import pandas as pd
import pprint


class Classifier:
    """ A simple Naive Bayes Classifier class.

    === Public Attributes ===
    dataset:
         The dataset provided to compute probabilities.
    class_atr:
         The class attribute used to calculate conditional probabilities.
    event:
         Event to find the conditional probability of with class_atr.
    priori:
         A dictionary of the naive probability of every outcome of class_attr in
         dataset.
    cp:
         Dictionary of conditional probabilities using bayes theorem of every
         class attribute (excluding class_attr) with class_attr.
    results:
         Dictionary of results for each outcome for class_atr after running the
         naive classification process.
    """
    # === Private Attributes ===
    # N/A
    # === Representation Invariants ===
    # - class_atr must be a valid class attribute in dataset.
    # - event must contain at least one class attribute found within dataset.
    dataset = None
    class_attr = None
    event = None
    priori = {}
    cp = {}
    results = {}

    def __init__(self, filename=None, class_attr=None, e=None) -> None:
        self.dataset = pd.read_csv(filename, sep=',', header=0)
        self.class_attr = class_attr
        self.event = e

    def calculate_priori(self) -> None:
        """
        Calculates the raw probabilities of every outcome of a class attribute;
        the priori probabilities.

        Probability attribute appears within class divided by total outcomes.
        """
        class_values = list(set(self.dataset[self.class_attr]))
        class_dataset = list(self.dataset[self.class_attr])
        for i in class_values:
            self.priori[i] = class_dataset.count(i) / len(class_dataset)

    def get_cp(self, attr, attr_type, class_value) -> float:
        """
        Probability of outcome (attr) given evidence (class_attr or inverse(s));
        Bayes theorem

        P(attr|class_attr) = P(class_attr|attr)P(attr)/P(class_attr)
        = (P(class_attr AND attr)/P(attr))P(attr)/P(class_attr)
        = P(class_attr AND attr)/P(class_attr)
        = |class_attr AND attr|/|S| / |class_attr|/|S|
        = |class_attr AND attr|/|class_attr|
        """
        dataset_attr = list(self.dataset[attr])
        class_dataset = list(self.dataset[self.class_attr])
        ands = 1
        for i in range(0, len(dataset_attr)):
            if class_dataset[i] == class_value and dataset_attr[i] == attr_type:
                ands += 1
        return ands / class_dataset.count(class_value)

    def calculate_conditional_probabilities(self) -> None:
        """
        Calculates using Bayes Theorem for every dataset attribute besides
        class_attr and its inverse(s).
        """
        for i in self.priori:
            self.cp[i] = {}
            for j in self.event:
                self.cp[i].update(
                    {self.event[j]: self.get_cp(j, self.event[j], i)})

    def classify(self) -> None:
        """
        Multiplies all conditional probabilities of an outcome of class_attr
        with the priori of each class_attr.
        """
        for i in self.cp:
            self.results[i] = {(reduce(lambda x, y: x * y, self.cp[i].values())
                               * self.priori[i])}

    def calculate(self) -> None:
        """
        Calculates then prints all information for the classifier in an
        organized manner.
        """
        self.calculate_priori()
        self.calculate_conditional_probabilities()
        self.classify()

        final = 0
        resulting = None
        print("Priori Values: " + str(self.priori) + "\n")
        print("Conditional Probabilities: ")
        pprint.pprint(self.cp)
        print("\nResult: ")
        for i in self.results:
            x = list(self.results[i])
            print(i + " => " + str(x))
            if x[0] > final:
                final = x[0]
                resulting = i
        print("\n" + self.class_attr + ": " + resulting)


if __name__ == "__main__":
    c = Classifier("Data.csv", "Go Outside?",
                   {"Weather": 'Sunny', "Temperature": 'Mild',
                    "Humidity": 'High', "Windy": 't'})
    c.calculate()
