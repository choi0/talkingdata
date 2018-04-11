#!/usr/bin/env python3
"""
Module Docstring
"""
import csv
import sys
import pandas as pd
#import plotly
import plotly.plotly as py
import plotly.graph_objs as go
#py.tools.set_credentials_file(username='hi', api_key='DONT LOOK AT MEEEE')

# Create random data with numpy
import numpy as np


__author__ = "Danny Choi"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    dataPoint = [];
    print("hello world")
    f = open("train_sample.csv", 'rt')
    reader = csv.reader(f, delimiter=',')
    for x in range(0, 3):
        currLine = next(reader)
        print(currLine)
        for index in range(len(currLine)):
            print(currLine[index])

    f.close()

    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)

    # Create a trace
    trace = go.Scatter(
        x = random_x,
        y = random_y,
        mode = 'markers'
    )

    data = [trace]

    df = pd.read_csv('train_sample_10000.csv')

    dataPointsAtt = df[(df.is_attributed==1)]
    dataPointsNonAtt = df[(df.is_attributed==0)]
    trace_comp0 = go.Scatter(
        x=dataPointsAtt.click_time,
        y=dataPointsAtt.channel,
        mode='markers',
        marker=dict(size=12,
                    line=dict(width=1),
                    color="navy"
                   ),
        name='dataPointsAtt',
        text=dataPointsAtt.ip,
        )

    trace_comp1 = go.Scatter(
        x=dataPointsNonAtt.click_time,
        y=dataPointsNonAtt.channel,
        mode='markers',
        marker=dict(size=12,
                    line=dict(width=1),
                    color="red"
                   ),
        name='dataPointsNonAtt',
        text=dataPointsNonAtt.ip,
            )

    data_comp = [trace_comp0, trace_comp1]
    layout_comp = go.Layout(
        title='Channel vs Time',
        hovermode='closest',
        xaxis=dict(
            title='Time',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Channel',
            ticklen=5,
            gridwidth=2,
        ),
    )
    fig_comp = go.Figure(data=data_comp, layout=layout_comp)
    py.iplot(fig_comp, filename='Channel vs Time')

    # Plot and embed in ipython notebook!
    #py.iplot(data, filename='basic-scatter')

    # or plot with: plot_url = py.plot(data, filename='basic-line')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
