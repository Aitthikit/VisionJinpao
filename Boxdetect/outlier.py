import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from scipy.spatial import distance


# After detecting outliers, call the plotting function


# Your dataset
data = [[[['red', [-303.95, 277.89, 903]], ['green', [-26.0, 277.0, 897]], ['blue', [278.17, 287.69, 920]]], [['blue', [-335.74, 88.0, 946]], ['green', [-30.72, 84.23, 894]], ['green', [275.0, 87.59, 918]]], [['red', [-314.96, -95.9, 887]], ['red', [-33.69, -104.05, 874]], ['blue', [296.15, -110.0, 900]]]], [[['red', [-302.26, 274.52, 894]], ['green', [-26.24, 279.54, 895]], ['blue', [268.32, 279.63, 907]]], [['blue', [-338.68, 87.81, 940]], ['green', [-30.72, 84.23, 899]], ['green', [275.0, 87.59, 917]]], [['red', [-311.36, -95.08, 893]], ['red', [-33.69, -103.06, 870]], ['blue', [296.15, -110.0, 902]]]], [[['red', [-302.26, 274.52, 912]], ['green', [-26.0, 277.0, 900]], ['blue', [268.32, 280.65, 920]]], [['blue', [-335.74, 88.0, 946]], ['green', [-31.0, 85.0, 904]], ['green', [277.57, 88.41, 918]]], [['red', [-311.36, -95.08, 886]], ['red', [-33.69, -104.05, 872]], ['blue', [291.6, -107.92, 893]]]], [[['red', [-302.26, 274.52, 898]], ['green', [-26.24, 279.54, 897]], ['blue', [268.32, 279.63, 931]]], [['blue', [-335.74, 88.0, 0]], ['green', [-31.0, 85.0, 898]], ['green', [277.57, 88.41, 925]]], [['red', [-314.96, -95.9, 887]], ['red', [-33.69, -103.06, 874]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-303.95, 276.93, 896]], ['green', [-26.24, 279.54, 897]], ['blue', [268.32, 279.63, 921]]], [['blue', [-335.74, 88.0, 960]], ['green', [-30.72, 84.23, 897]], ['green', [277.57, 88.41, 920]]], [['red', [-314.96, -95.9, 885]], ['red', [-32.7, -103.06, 879]], ['blue', [293.33, -108.95, 890]]]], [[['red', [-302.26, 274.52, 910]], ['green', [-26.0, 276.0, 896]], ['blue', [268.32, 280.65, 933]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 896]], ['green', [277.57, 88.41, 921]]], [['red', [-314.96, -95.9, 894]], ['red', [-32.7, -103.06, 873]], ['blue', [291.6, -106.89, 904]]]], [[['red', [-302.26, 274.52, 912]], ['green', [-26.24, 279.54, 893]], ['blue', [268.32, 280.65, 922]]], [['blue', [-335.74, 88.0, 0]], ['green', [-30.72, 84.23, 889]], ['green', [277.57, 88.41, 921]]], [['red', [-314.96, -95.9, 893]], ['red', [-32.41, -102.14, 870]], ['blue', [291.6, -106.89, 900]]]], [[['red', [-302.26, 274.52, 890]], ['green', [-26.24, 279.54, 897]], ['blue', [268.32, 279.63, 924]]], [['blue', [-335.74, 87.04, 937]], ['green', [-30.72, 84.23, 894]], ['green', [281.23, 89.25, 920]]], [['red', [-309.66, -94.29, 892]], ['red', [-33.1, -101.24, 874]], ['blue', [291.6, -107.92, 911]]]], [[['red', [-302.26, 274.52, 913]], ['green', [-26.24, 278.53, 902]], ['blue', [268.32, 279.63, 924]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 900]], ['green', [277.57, 88.41, 919]]], [['red', [-312.29, -95.08, 883]], ['red', [-32.7, -103.06, 876]], ['blue', [291.6, -107.92, 899]]]], [[['red', [-299.66, 272.16, 893]], ['green', [-26.48, 281.11, 906]], ['blue', [268.32, 279.63, 911]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 895]], ['green', [277.57, 89.44, 925]]], [['red', [-311.36, -95.08, 895]], ['red', [-32.41, -102.14, 872]], ['blue', [291.6, -107.92, 907]]]], [[['red', [-299.66, 272.16, 899]], ['green', [-26.48, 281.11, 900]], ['blue', [268.32, 279.63, 922]]], [['blue', [-335.74, 87.04, 933]], ['green', [-30.72, 84.23, 904]], ['green', [277.57, 88.41, 916]]], [['red', [-312.29, -95.08, 894]], ['red', [-32.41, -102.14, 873]], ['blue', [291.6, -107.92, 908]]]], [[['red', [-303.22, 274.52, 900]], ['green', [-26.48, 282.13, 900]], ['blue', [268.32, 279.63, 916]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 904]], ['green', [277.57, 88.41, 918]]], [['red', [-311.36, -95.08, 887]], ['red', [-33.1, -101.24, 885]], ['blue', [291.6, -107.92, 896]]]], [[['red', [-302.26, 274.52, 897]], ['green', [-26.24, 282.57, 902]], ['blue', [268.32, 279.63, 920]]], [['blue', [-335.74, 87.04, 942]], ['green', [-30.72, 84.23, 905]], ['green', [277.57, 89.44, 920]]], [['red', [-314.96, -95.9, 901]], ['red', [-33.1, -101.24, 875]], ['blue', [291.6, -107.92, 896]]]], [[['red', [-303.22, 274.52, 903]], ['green', [-26.24, 283.58, 914]], ['blue', [268.32, 279.63, 925]]], [['blue', [-335.74, 87.04, 932]], ['green', [-30.45, 83.48, 900]], ['green', [277.57, 88.41, 918]]], [['red', [-309.66, -94.29, 891]], ['red', [-32.41, -102.14, 869]], ['blue', [291.6, -107.92, 898]]]], [[['red', [-302.26, 274.52, 900]], ['green', [-25.0, 280.0, 897]], ['blue', [268.32, 279.63, 931]]], [['blue', [-337.72, 87.81, 0]], ['green', [-30.72, 84.23, 896]], ['green', [277.57, 89.44, 915]]], [['red', [-314.96, -95.9, 893]], ['red', [-32.41, -102.14, 876]], ['blue', [291.6, -107.92, 899]]]], [[['red', [-299.66, 272.16, 897]], ['green', [-26.24, 283.58, 901]], ['blue', [270.85, 282.26, 915]]], [['blue', [-335.74, 87.04, 948]], ['green', [-30.72, 84.23, 888]], ['green', [277.57, 89.44, 916]]], [['red', [-309.66, -94.29, 884]], ['red', [-32.41, -102.14, 872]], ['blue', [291.6, -107.92, 896]]]], [[['red', [-302.26, 274.52, 901]], ['green', [-26.48, 281.11, 895]], ['blue', [268.32, 279.63, 920]]], [['blue', [-335.74, 87.04, 940]], ['green', [-31.0, 85.0, 889]], ['green', [277.57, 89.44, 920]]], [['red', [-311.36, -95.08, 895]], ['red', [-33.1, -101.24, 875]], ['blue', [291.6, -108.96, 900]]]], [[['red', [-303.22, 274.52, 894]], ['green', [-26.24, 278.53, 916]], ['blue', [268.32, 279.63, 927]]], [['blue', [-335.74, 87.04, 945]], ['green', [-30.72, 84.23, 908]], ['green', [277.57, 88.41, 921]]], [['red', [-311.36, -95.08, 894]], ['red', [-32.41, -102.14, 867]], ['blue', [291.6, -107.92, 896]]]], [[['red', [-299.66, 272.16, 897]], ['green', [-26.24, 282.57, 909]], ['blue', [270.85, 282.26, 916]]], [['blue', [-335.74, 87.04, 936]], ['green', [-30.72, 84.23, 905]], ['green', [280.19, 89.25, 919]]], [['red', [-309.66, -94.29, 907]], ['red', [-33.1, -101.24, 882]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-302.26, 274.52, 915]], ['green', [-25.0, 280.0, 893]], ['blue', [270.85, 282.26, 916]]], [['blue', [-335.74, 87.04, 942]], ['green', [-31.0, 85.0, 900]], ['green', [282.86, 90.1, 916]]], [['red', [-312.29, -95.08, 893]], ['red', [-32.7, -103.06, 875]], ['blue', [291.6, -108.96, 894]]]], [[['red', [-304.91, 276.93, 920]], ['green', [-26.48, 281.11, 900]], ['blue', [268.32, 279.63, 916]]], [['blue', [-338.68, 87.81, 948]], ['green', [-30.72, 83.24, 893]], ['green', [277.57, 89.44, 916]]], [['red', [-309.66, -94.29, 891]], ['red', [-32.7, -104.05, 879]], ['blue', [291.6, -108.96, 904]]]], [[['red', [-304.91, 276.93, 905]], ['green', [-26.48, 281.11, 899]], ['blue', [270.85, 282.26, 923]]], [['blue', [-335.74, 87.04, 0]], ['green', [-31.0, 85.0, 898]], ['green', [281.23, 89.25, 917]]], [['red', [-309.66, -94.29, 890]], ['red', [-32.7, -103.06, 874]], ['blue', [291.6, -108.96, 895]]]], [[['red', [-302.26, 274.52, 898]], ['green', [-26.24, 283.58, 902]], ['blue', [268.32, 279.63, 931]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 908]], ['green', [277.57, 88.41, 916]]], [['red', [-309.66, -94.29, 893]], ['red', [-33.1, -101.24, 879]], ['blue', [291.6, -107.92, 893]]]], [[['red', [-302.26, 274.52, 911]], ['green', [-26.48, 282.13, 896]], ['blue', [270.85, 282.26, 937]]], [['blue', [-335.74, 87.04, 936]], ['green', [-30.72, 84.23, 892]], ['green', [280.19, 89.25, 920]]], [['red', [-314.96, -95.9, 896]], ['red', [-32.41, -103.12, 877]], ['blue', [293.33, -108.95, 899]]]], [[['red', [-299.66, 272.16, 897]], ['green', [-26.48, 281.11, 902]], ['blue', [270.85, 282.26, 926]]], [['blue', [-335.74, 87.04, 936]], ['green', [-30.72, 84.23, 905]], ['green', [277.57, 88.41, 920]]], [['red', [-309.66, -94.29, 889]], ['red', [-32.41, -103.12, 867]], ['blue', [291.6, -107.92, 888]]]], [[['red', [-302.26, 274.52, 893]], ['green', [-26.48, 281.11, 903]], ['blue', [270.85, 282.26, 913]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 906]], ['green', [277.57, 88.41, 916]]], [['red', [-312.29, -95.08, 897]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 900]]]], [[['red', [-299.66, 272.16, 904]], ['green', [-26.48, 281.11, 905]], ['blue', [270.85, 282.26, 927]]], [['blue', [-335.74, 87.04, 949]], ['green', [-30.72, 83.24, 903]], ['green', [280.19, 90.28, 918]]], [['red', [-311.36, -95.08, 897]], ['red', [-32.41, -102.14, 872]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-299.66, 272.16, 895]], ['green', [-26.48, 282.13, 903]], ['blue', [270.85, 282.26, 920]]], [['blue', [-335.74, 87.04, 945]], ['green', [-30.72, 84.23, 891]], ['green', [277.57, 88.41, 925]]], [['red', [-311.36, -95.08, 893]], ['red', [-32.41, -102.14, 880]], ['blue', [291.6, -107.92, 899]]]], [[['red', [-299.66, 272.16, 902]], ['green', [-26.48, 281.11, 900]], ['blue', [270.85, 282.26, 925]]], [['blue', [-335.74, 86.09, 953]], ['green', [-30.72, 83.24, 896]], ['green', [280.19, 89.25, 917]]], [['red', [-309.66, -94.29, 888]], ['red', [-33.1, -102.21, 874]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-299.66, 272.16, 912]], ['green', [-26.24, 283.58, 903]], ['blue', [268.32, 279.63, 917]]], [['blue', [-335.74, 87.04, 939]], ['green', [-31.0, 85.0, 899]], ['green', [277.57, 88.41, 916]]], [['red', [-309.66, -94.29, 893]], ['red', [-32.41, -103.12, 865]], ['blue', [291.6, -107.92, 906]]]], [[['red', [-302.26, 274.52, 898]], ['green', [-26.24, 282.57, 908]], ['blue', [270.85, 282.26, 926]]], [['blue', [-335.74, 87.04, 947]], ['green', [-30.45, 83.48, 889]], ['green', [277.57, 88.41, 915]]], [['red', [-311.36, -95.08, 894]], ['red', [-32.41, -102.14, 878]], ['blue', [291.6, -107.92, 897]]]], [[['red', [-303.22, 274.52, 904]], ['green', [-26.24, 282.57, 903]], ['blue', [270.85, 282.26, 931]]], [['blue', [-335.74, 86.09, 946]], ['green', [-30.72, 84.23, 897]], ['green', [277.57, 88.41, 916]]], [['red', [-311.36, -95.08, 890]], ['red', [-32.41, -102.14, 874]], ['blue', [291.6, -107.92, 899]]]], [[['red', [-302.26, 274.52, 887]], ['green', [-26.48, 281.11, 898]], ['blue', [268.32, 280.65, 917]]], [['blue', [-335.74, 87.04, 946]], ['green', [-30.72, 84.23, 897]], ['green', [281.23, 89.25, 918]]], [['red', [-312.29, -95.08, 898]], ['red', [-33.1, -101.24, 872]], ['blue', [291.6, -107.92, 900]]]], [[['red', [-299.66, 272.16, 896]], ['green', [-26.48, 281.11, 896]], ['blue', [270.85, 282.26, 917]]], [['blue', [-335.74, 86.09, 948]], ['green', [-30.72, 84.23, 908]], ['green', [281.23, 89.25, 923]]], [['red', [-309.66, -94.29, 894]], ['red', [-32.41, -102.14, 878]], ['blue', [291.6, -107.92, 897]]]], [[['red', [-299.66, 272.16, 904]], ['green', [-26.48, 282.13, 906]], ['blue', [270.85, 282.26, 912]]], [['blue', [-335.74, 87.04, 931]], ['green', [-30.72, 84.23, 900]], ['green', [280.19, 89.25, 916]]], [['red', [-309.66, -94.29, 895]], ['red', [-33.1, -101.24, 874]], ['blue', [291.6, -107.92, 899]]]], [[['red', [-302.26, 274.52, 904]], ['green', [-26.48, 282.13, 897]], ['blue', [270.85, 282.26, 929]]], [['blue', [-335.74, 88.0, 942]], ['green', [-30.72, 83.24, 905]], ['green', [277.57, 88.41, 922]]], [['red', [-314.02, -95.9, 900]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -108.96, 909]]]], [[['red', [-303.22, 274.52, 901]], ['green', [-26.48, 282.13, 897]], ['blue', [268.32, 279.63, 922]]], [['blue', [-335.74, 87.04, 942]], ['green', [-30.72, 84.23, 905]], ['green', [280.19, 89.25, 923]]], [['red', [-311.36, -95.08, 900]], ['red', [-32.41, -102.14, 878]], ['blue', [291.6, -107.92, 900]]]], [[['red', [-299.66, 272.16, 897]], ['green', [-26.48, 282.13, 900]], ['blue', [270.85, 282.26, 928]]], [['blue', [-335.74, 87.04, 952]], ['green', [-30.72, 83.24, 897]], ['green', [277.57, 88.41, 916]]], [['red', [-309.66, -94.29, 878]], ['red', [-32.7, -104.05, 873]], ['blue', [291.6, -107.92, 893]]]], [[['red', [-299.66, 272.16, 899]], ['green', [-26.24, 282.57, 897]], ['blue', [268.32, 279.63, 918]]], [['blue', [-335.74, 87.04, 947]], ['green', [-30.72, 84.23, 893]], ['green', [277.57, 88.41, 920]]], [['red', [-312.29, -95.08, 893]], ['red', [-32.41, -102.14, 886]], ['blue', [291.6, -107.92, 892]]]], [[['red', [-303.22, 274.52, 897]], ['green', [-26.48, 281.11, 904]], ['blue', [268.32, 280.65, 918]]], [['blue', [-335.74, 87.04, 936]], ['green', [-30.72, 84.23, 900]], ['green', [277.57, 88.41, 920]]], [['red', [-311.36, -95.08, 887]], ['red', [-32.41, -102.14, 874]], ['blue', [291.6, -107.92, 903]]]], [[['red', [-299.66, 272.16, 907]], ['green', [-26.24, 283.58, 904]], ['blue', [270.85, 282.26, 912]]], [['blue', [-335.74, 87.04, 948]], ['green', [-30.72, 84.23, 900]], ['green', [277.57, 88.41, 919]]], [['red', [-311.36, -95.08, 890]], ['red', [-32.41, -102.14, 874]], ['blue', [291.6, -108.96, 896]]]], [[['red', [-299.66, 272.16, 900]], ['green', [-26.48, 282.13, 900]], ['blue', [268.32, 279.63, 933]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 900]], ['green', [277.57, 88.41, 919]]], [['red', [-312.29, -95.08, 893]], ['red', [-33.1, -101.24, 880]], ['blue', [291.6, -108.96, 903]]]], [[['red', [-299.66, 272.16, 898]], ['green', [-26.48, 281.11, 904]], ['blue', [268.32, 279.63, 927]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.28, 85.78, 904]], ['green', [277.57, 89.44, 917]]], [['red', [-312.29, -95.08, 890]], ['red', [-32.41, -103.12, 884]], ['blue', [291.6, -106.89, 895]]]], [[['red', [-299.66, 272.16, 898]], ['green', [-26.24, 282.57, 896]], ['blue', [270.85, 282.26, 916]]], [['blue', [-337.72, 86.84, 1000]], ['green', [-30.72, 84.23, 900]], ['green', [280.19, 89.25, 912]]], [['red', [-314.96, -95.9, 887]], ['red', [-32.41, -102.14, 867]], ['blue', [291.6, -107.92, 900]]]], [[['red', [-299.66, 272.16, 900]], ['green', [-26.48, 282.13, 898]], ['blue', [270.85, 282.26, 916]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 907]], ['green', [277.57, 88.41, 916]]], [['red', [-312.29, -95.08, 900]], ['red', [-33.1, -101.24, 873]], ['blue', [291.6, -107.92, 902]]]], [[['red', [-302.26, 274.52, 904]], ['green', [-26.48, 282.13, 897]], ['blue', [268.32, 279.63, 928]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.0, 85.0, 895]], ['green', [277.57, 88.41, 921]]], [['red', [-312.29, -95.08, 877]], ['red', [-32.41, -102.14, 877]], ['blue', [291.6, -107.92, 906]]]], [[['red', [-299.66, 272.16, 896]], ['green', [-26.48, 282.13, 903]], ['blue', [270.85, 282.26, 935]]], [['blue', [-335.74, 86.09, 940]], ['green', [-30.0, 85.0, 906]], ['green', [282.86, 90.1, 920]]], [['red', [-309.66, -94.29, 898]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 901]]]], [[['red', [-299.66, 272.16, 893]], ['green', [-25.0, 281.0, 905]], ['blue', [270.85, 283.3, 910]]], [['blue', [-335.74, 87.04, 940]], ['green', [-31.0, 85.0, 902]], ['green', [277.57, 88.41, 912]]], [['red', [-314.96, -95.9, 890]], ['red', [-32.41, -102.14, 876]], ['blue', [291.6, -108.96, 901]]]], [[['red', [-302.26, 274.52, 889]], ['green', [-26.48, 282.13, 897]], ['blue', [268.32, 279.63, 925]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 895]], ['green', [277.57, 89.44, 919]]], [['red', [-311.36, -95.08, 895]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-304.91, 276.93, 897]], ['green', [-26.48, 281.11, 897]], ['blue', [268.32, 279.63, 923]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.0, 85.0, 904]], ['green', [277.57, 89.44, 920]]], [['red', [-309.66, -94.29, 895]], ['red', [-32.41, -102.14, 877]], ['blue', [291.6, -107.92, 900]]]], [[['red', [-302.26, 274.52, 891]], ['green', [-26.48, 282.13, 906]], ['blue', [268.32, 279.63, 912]]], [['blue', [-335.74, 87.04, 943]], ['green', [-31.0, 85.0, 900]], ['green', [280.19, 89.25, 914]]], [['red', [-312.29, -95.08, 896]], ['red', [-33.1, -101.24, 876]], ['blue', [293.33, -108.95, 897]]]], [[['red', [-299.66, 272.16, 890]], ['green', [-26.48, 282.13, 903]], ['blue', [268.32, 279.63, 925]]], [['blue', [-335.74, 87.04, 0]], ['green', [-31.0, 85.0, 905]], ['green', [277.57, 88.41, 924]]], [['red', [-312.29, -95.08, 896]], ['red', [-32.41, -103.12, 879]], ['blue', [291.6, -106.89, 897]]]], [[['red', [-302.26, 274.52, 912]], ['green', [-26.48, 281.11, 902]], ['blue', [268.32, 279.63, 922]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 895]], ['green', [277.57, 88.41, 916]]], [['red', [-311.36, -95.08, 894]], ['red', [-33.1, -101.24, 875]], ['blue', [291.6, -107.92, 889]]]], [[['red', [-303.22, 274.52, 904]], ['green', [-26.24, 278.53, 903]], ['blue', [268.32, 279.63, 923]]], [['blue', [-335.74, 87.04, 943]], ['green', [-31.28, 85.78, 902]], ['green', [280.19, 88.21, 923]]], [['red', [-312.29, -95.08, 900]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 903]]]], [[['red', [-299.66, 272.16, 891]], ['green', [-26.24, 282.57, 908]], ['blue', [268.32, 279.63, 925]]], [['blue', [-335.74, 87.04, 940]], ['green', [-31.0, 85.0, 900]], ['green', [280.19, 90.28, 916]]], [['red', [-309.66, -94.29, 888]], ['red', [-32.41, -102.14, 874]], ['blue', [291.6, -108.96, 903]]]], [[['red', [-299.66, 272.16, 908]], ['green', [-26.48, 282.13, 905]], ['blue', [270.85, 282.26, 916]]], [['blue', [-335.74, 87.04, 942]], ['green', [-30.72, 84.23, 905]], ['green', [281.23, 90.28, 920]]], [['red', [-311.36, -95.08, 893]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 897]]]], [[['red', [-299.66, 272.16, 894]], ['green', [-26.48, 281.11, 902]], ['blue', [270.85, 283.3, 928]]], [['blue', [-335.74, 87.04, 942]], ['green', [-30.72, 84.23, 897]], ['green', [277.57, 88.41, 915]]], [['red', [-311.36, -95.08, 887]], ['red', [-33.1, -101.24, 879]], ['blue', [291.6, -107.92, 903]]]], [[['red', [-299.66, 272.16, 887]], ['green', [-26.48, 281.11, 903]], ['blue', [270.85, 282.26, 925]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 895]], ['green', [280.19, 90.28, 921]]], [['red', [-309.66, -94.29, 885]], ['red', [-32.41, -103.12, 875]], ['blue', [293.33, -108.95, 897]]]], [[['red', [-303.22, 274.52, 904]], ['green', [-26.48, 282.13, 901]], ['blue', [270.85, 283.3, 915]]], [['blue', [-338.68, 87.81, 939]], ['green', [-31.0, 85.0, 896]], ['green', [277.57, 89.44, 917]]], [['red', [-309.66, -94.29, 890]], ['red', [-32.41, -103.12, 872]], ['blue', [291.6, -107.92, 907]]]], [[['red', [-304.91, 276.93, 904]], ['green', [-26.24, 278.53, 900]], ['blue', [270.85, 282.26, 930]]], [['blue', [-335.74, 87.04, 936]], ['green', [-31.0, 85.0, 908]], ['green', [276.54, 88.41, 918]]], [['red', [-312.29, -95.08, 893]], ['red', [-32.41, -102.14, 881]], ['blue', [291.6, -107.92, 894]]]], [[['red', [-304.91, 276.93, 901]], ['green', [-25.0, 280.0, 905]], ['blue', [268.32, 279.63, 930]]], [['blue', [-335.74, 87.04, 955]], ['green', [-31.0, 85.0, 891]], ['green', [280.19, 90.28, 914]]], [['red', [-309.66, -94.29, 897]], ['red', [-32.41, -102.14, 877]], ['blue', [291.6, -107.92, 905]]]], [[['red', [-303.22, 274.52, 919]], ['green', [-26.48, 281.11, 900]], ['blue', [270.85, 282.26, 916]]], [['blue', [-335.74, 87.04, 940]], ['green', [-30.72, 84.23, 900]], ['green', [277.57, 88.41, 919]]], [['red', [-309.66, -94.29, 893]], ['red', [-32.41, -102.14, 872]], ['blue', [291.6, -107.92, 902]]]], [[['red', [-299.66, 272.16, 902]], ['green', [-26.24, 282.57, 900]], ['blue', [270.85, 283.3, 926]]], [['blue', [-335.74, 87.04, 945]], ['green', [-30.72, 84.23, 908]], ['green', [277.57, 88.41, 921]]], [['red', [-312.29, -95.08, 906]], ['red', [-32.41, -102.14, 872]], ['blue', [291.6, -107.92, 903]]]], [[['red', [-299.66, 272.16, 906]], ['green', [-26.48, 281.11, 906]], ['blue', [270.85, 282.26, 925]]], [['blue', [-335.74, 87.04, 945]], ['green', [-31.0, 85.0, 893]], ['green', [280.19, 89.25, 917]]], [['red', [-309.66, -94.29, 890]], ['red', [-33.1, -101.24, 878]], ['blue', [291.6, -107.92, 897]]]], [[['red', [-302.26, 274.52, 900]], ['green', [-26.48, 281.11, 901]], ['blue', [270.85, 282.26, 923]]], [['blue', [-335.74, 87.04, 945]], ['green', [-30.72, 84.23, 907]], ['green', [277.57, 88.41, 920]]], [['red', [-309.66, -94.29, 898]], ['red', [-33.1, -101.24, 878]], ['blue', [291.6, -107.92, 906]]]], [[['red', [-303.22, 274.52, 902]], ['green', [-25.0, 281.0, 900]], ['blue', [270.85, 282.26, 920]]], [['blue', [-335.74, 86.09, 942]], ['green', [-30.72, 84.23, 898]], ['green', [280.19, 89.25, 915]]], [['red', [-311.36, -95.08, 890]], ['red', [-32.41, -102.14, 875]], ['blue', [291.6, -107.92, 898]]]], [[['red', [-299.66, 272.16, 894]], ['green', [-26.48, 281.11, 902]], ['blue', [270.85, 282.26, 914]]], [['blue', [-335.74, 87.04, 0]], ['green', [-30.72, 84.23, 904]], ['green', [277.57, 88.41, 924]]], [['red', [-311.36, -95.08, 899]], ['red', [-32.41, -102.14, 878]], ['blue', [291.6, -107.92, 904]]]], [[['red', [-303.22, 274.52, 897]], ['green', [-26.48, 282.13, 897]], ['blue', [268.32, 279.63, 926]]], [['blue', [-338.68, 87.81, 937]], ['green', [-31.0, 85.0, 898]], ['green', [277.57, 88.41, 919]]], [['red', [-309.66, -94.29, 900]], ['red', [-32.41, -103.12, 878]], ['blue', [291.6, -107.92, 900]]]]]
    # ... (your data goes here, truncated for brevity)

def flatten_data(data):
    return np.array([point[1] for sublist in data for subsublist in sublist for point in subsublist if isinstance(point[1], list)])

# Function to calculate Euclidean distance for each point in the dataset
def calculate_euclidean(data):
    data_mean = np.mean(data, axis=0)
    euclidean_distances = np.array([distance.euclidean(point, data_mean) for point in data])
    return euclidean_distances

# Function to identify outliers using Euclidean distance
def identify_outliers_euclidean(data, threshold):
    distances = calculate_euclidean(data)
    return data[distances > threshold]

# Function to plot the data
def plot_data_with_outliers(data, outliers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting all data points
    ax.scatter(data[:,0], data[:,1], data[:,2], c='blue', label='Raw Data')

    # Highlighting outliers, if any
    if len(outliers) > 0:
        ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], c='red', label='Outliers')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    plt.show()


def plot_3d_scatter(data, x_label='X Axis', y_label='Y Axis', z_label='Z Axis', title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting all data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', label='Raw Data')

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Add a legend
    ax.legend()

    # Set plot title if provided
    if title:
        plt.title(title)

    # Show the plot
    plt.show()
# Flattening the data to get only the position coordinates


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks



flat_data = flatten_data(data)
# print(flat_data[:,0])
import seaborn as sns  
# Create a KDE plot using Seaborn
sns.kdeplot(flat_data[:,0], shade=True)

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('KDE Plot of Data Distribution')

# Compute the KDE values
kde_values_x = sns.kdeplot(flat_data[:,0]).get_lines()[0].get_data()
kde_values_y = sns.kdeplot(flat_data[:,1]).get_lines()[1].get_data()
kde_values_z = sns.kdeplot(flat_data[:,2]).get_lines()[2].get_data()


# print(kde_values)

# Find multiple peaks_x using scipy's find_peaks_x
peaks_x, _ = find_peaks(kde_values_x[1], height=0)  # Adjust the height threshold as needed
peaks_y, _ = find_peaks(kde_values_y[1], height=0)  # Adjust the height threshold as needed
peaks_z, _ = find_peaks(kde_values_z[1], height=0)  # Adjust the height threshold as needed


# Get the positions and values of the peaks_x
peak_positions_x = (kde_values_x[0][peaks_x])
peak_positions_y = kde_values_y[0][peaks_y][::-1]
peak_positions_z = kde_values_z[0][peaks_z]
print(max(peak_positions_z))

# peak_values_x = (kde_values_x[1][peaks_x])
# peak_values_y = kde_values_y[1][peaks_y]
# print(peak_positions)
# print(peak_positions2)
for i in range(len(peak_positions_x)):
    # print(peak_positions[i])
    for k in range(len(peak_positions_y)):
        print([peak_positions_x[k],peak_positions_y[i],max(peak_positions_z),[i,k]])

# Plot the peaks_x as red dots
# plt.plot(peak_positions2, peak_values2, 'ro', markersize=8)

plt.show()
# plot_data_with_outliers(flat_data,flat_data)
# plt.scatter(flatten_data[:,0],flatten_data[:,1],flatten_data[:,2], c='blue', label='Raw Data')
