#!/usr/bin/env python3
"""
Module Docstring
"""
import csv
import sys

__author__ = "Danny Choi"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    print("hello world")
    f = open("train_sample.csv", 'rt')
    reader = csv.reader(f, delimiter=',')
    for x in range(0, 3):
        currLine = next(reader)
        print(currLine)
        for index in range(len(currLine)):
            print(currLine[index])

    f.close()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
