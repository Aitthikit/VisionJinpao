tt = [[[['green', [-14.26, 69.26, 1699]], ['blue', [126.76, 139.33, 1707]], ['blue', [615.19, 67.22, 1815]]], [['blue', [-23.33, -202.22, 1693]], ['blue', [302.0, -136.0, 1699]], ['green', [638.43, -135.88, 1713]]], [['blue', [-240.62, -1165.31, 1642]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-16.0, 68.0, 1642]], ['blue', [111.68, 110.84, 1707]], ['blue', [624.72, 72.64, 1754]]], [['blue', [-47.71, -234.58, 1707]], ['blue', [302.0, -132.0, 1664]], ['green', [638.43, -138.04, 1778]]], [['blue', [-240.62, -1165.31, 1642]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-14.26, 69.26, 1722]], ['red', [295.37, 75.37, 1691]], ['blue', [615.19, 65.19, 0]]], [['blue', [-25.21, -222.29, 1710]], ['blue', [302.0, -128.0, 1745]], ['green', [638.43, -138.04, 1693]]], [['blue', [-240.62, -1165.31, 1642]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-13.75, 66.79, 1640]], ['blue', [113.49, 114.37, 1719]], ['blue', [615.19, 63.15, 1727]]], [['blue', [-24.33, -190.38, 1682]], ['blue', [302.0, -130.0, 1693]], ['green', [638.43, -135.88, 1745]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-14.26, 69.26, 1680]], ['blue', [126.76, 137.24, 1699]], ['blue', [615.19, 71.3, 1730]]], [['blue', [-35.0, -242.5, 1666]], ['blue', [302.0, -132.0, 1656]], ['green', [626.15, -135.38, 1650]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-14.26, 69.26, 1688]], ['blue', [126.76, 136.19, 1696]], ['blue', [615.19, 67.22, 1760]]], [['blue', [-24.18, -234.51, 1661]], ['blue', [302.0, -130.0, 1699]], ['green', [638.43, -138.04, 1739]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-14.0, 70.0, 1680]], ['blue', [125.57, 146.32, 1707]], ['blue', [615.19, 69.26, 1796]]], [['blue', [-33.12, -212.9, 1713]], ['blue', [302.0, -136.0, 1699]], ['green', [638.43, -140.2, 1724]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-14.26, 69.26, 1674]], ['blue', [257.87, 250.66, 1730]], ['blue', [615.19, 73.33, 1730]]], [['blue', [-24.91, -201.32, 1645]], ['blue', [302.0, -128.0, 1619]], ['green', [638.43, -138.04, 1781]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']], [[['green', [-13.75, 66.79, 1693]], ['blue', [129.04, 144.9, 1716]], ['blue', [615.19, 71.3, 1781]]], [['blue', [-29.79, -224.58, 1658]], ['blue', [302.0, -130.0, 1730]], ['green', [638.43, -138.04, 1739]]], [['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']]]
[['blue', [-240.62, -1165.31, 1642]], ['green', [770.0, -929.5, 1724]], '']
[['blue', [-240.62, -1165.31, 1642]], ['green', [770.0, -929.5, 1724]], '']
[['green', [-35.96, -393.46, 1693]], ['green', [770.0, -929.5, 1724]], '']
import numpy as np
kk = []
for i in tt:
    print("sad")
    # print(i)
    for j in i:
        print(j[2])
        print(type(j[2]))
        print(len(j))

    # i = np.array(i).reshape(-1)
    
    # if '' not in i:
    #     kk.append(i)
    #     # print(i)
# print(kk)