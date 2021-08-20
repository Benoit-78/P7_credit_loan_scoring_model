# Author: B.Delorme
# Creation date: 20/07/2021
# Main objective: to provide a support for correlations visualisation.

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics as stat

from std_q7 import QualityTool


class CorrelationDiagram(QualityTool):
    '''
    '''
    def __init__(self, data):
        '''
        Save the input data as arguments of the class.
        '''
        self.data = pd.DataFrame(data)
        self.data = self.data.dropna()
        self.list1 = self.data.iloc[:, 0]
        self.list2 = self.data.iloc[:, 1]


    def clean_data(self):
        '''

        '''


    def keypoints_coordinates(self, list1, list2):
        '''
        Given two lists, compute the average value, the standard deviation.
        Returns the coordinates of the four points that will form the cross,
        plus the coordinates of the intersection point.
        '''
        # Get the data
        med_1 = stat.median(list1)
        std_1 = stat.stdev(list1)
        med_2 = stat.median(list2)
        std_2 = stat.stdev(list2)
        # Set the points coordinates
        left_point = (med_1 - 3 * std_1, med_2)
        right_point = (med_1 + 3 * std_1, med_2)
        down_point = (med_1, med_2 - 3 * std_2)
        up_point = (med_1, med_2 + 3 * std_2)
        central_point = (med_1, med_2)
        # Return the results
        return left_point, right_point, down_point, up_point, central_point


    def plot(self):
        '''

        '''
        #sns.regplot(self.list1, self.list2)
        #sns.relplot(x=self.list1, y=self.list2, hue = self.list1)
        g = sns.jointplot(self.list1, self.list2, kind="reg")
        g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
        #l_point, r_point, d_point, u_point, c_point = self.keypoints_coordinates(
        #    self.list1, self.list2)
        #hori_x_values = [l_point[0], r_point[0]]
        #hori_y_values = [l_point[1], r_point[1]]
        #vert_x_values = [d_point[0], u_point[0]]
        #vert_y_values = [d_point[1], u_point[1]]
        #plt.plot(hori_x_values, hori_y_values, color='orange')
        #plt.plot(vert_x_values, vert_y_values, color='c')
        # Identify the zones
        #plt.text(x=r_point[0]*0.9, y=u_point[1]*0.9, s='I',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=5.0))
        #plt.text(x=r_point[0]*0.9, y=d_point[1]*1.1, s='II',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=5.0))
        #plt.text(x=l_point[0]*0.9, y=d_point[1]*1.1, s='III',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=5.0))
        #plt.text(x=l_point[0]*0.9, y=u_point[1]*0.9, s='IV',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=5.0))