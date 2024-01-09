import math
data = [[604, 272, 1022, 'Red'], [715, 271, 1022, 'Red'], [602, 154, 1021, 'Red'], [600, 389, 1025, 'Green'], [404, 154, 1037, 'Green'], [774, 152, 1017, 'Green'], [767, 390, 1018, 'Blue'], [345, 388, 1018, 'Blue'], [350, 271, 1018, 'Blue'], [603, 272, 1032, 'Red'], [715, 271, 1032, 'Red'], [602, 154, 1016, 'Red'], [600, 389, 1021, 'Green'], [405, 154, 1031, 'Green'], [774, 151, 1021, 'Green'], [766, 391, 1010, 'Blue'], [345, 389, 1010, 'Blue'], [350, 271, 1010, 'Blue'], [603, 272, 1022, 'Red'], [716, 271, 1022, 'Red'], [602, 154, 1026, 'Red'], [600, 390, 1022, 'Green'], [404, 153, 1048, 'Green'], [774, 151, 1018, 'Green'], [766, 391, 997, 'Blue'], [345, 389, 997, 'Blue'], [351, 271, 997, 'Blue'], [603, 272, 1031, 'Red'], [715, 271, 1031, 'Red'], [602, 154, 1022, 'Red'], [601, 389, 1021, 'Green'], [405, 153, 1032, 'Green'], [774, 151, 1018, 'Green'], [766, 391, 1019, 'Blue'], [344, 388, 1019, 'Blue'], [350, 270, 1019, 'Blue'], [604, 272, 1024, 'Red'], [715, 271, 1024, 'Red'], [602, 154, 1024, 'Red'], [600, 389, 1022, 'Green'], [404, 153, 1041, 'Green'], [774, 151, 1014, 'Green'], [767, 390, 1010, 'Blue'], [345, 388, 1010, 'Blue'], [350, 271, 1010, 'Blue'], [604, 272, 1023, 'Red'], [717, 271, 1023, 'Red'], [602, 154, 1027, 'Red'], [600, 389, 1022, 'Green'], [405, 153, 1047, 'Green'], [774, 151, 1013, 'Green'], [767, 390, 1029, 'Blue'], [344, 387, 1029, 'Blue'], [351, 271, 1029, 'Blue'], [605, 272, 1024, 'Red'], [715, 271, 1024, 'Red'], [602, 154, 1024, 'Red'], [600, 389, 1021, 'Green'], [404, 154, 1055, 'Green'], [774, 151, 1016, 'Green'], [766, 390, 1017, 'Blue'], [345, 387, 1017, 'Blue'], [350, 271, 1017, 'Blue'], [604, 272, 1014, 'Red'], [715, 271, 1014, 'Red'], [602, 154, 1022, 'Red'], [600, 389, 1017, 'Green'], [404, 154, 1037, 'Green'], [774, 152, 1015, 'Green'], [767, 390, 1019, 'Blue'], [344, 388, 1019, 'Blue'], [350, 270, 1019, 'Blue']]

def find_table(data):
    temp = [["","",""],["","",""],["","",""]]
    output = [["","",""],["","",""],["","",""]]
    check = []
    min_x = min_x = min(point[0] for point in data)
    max_x = max(point[0] for point in data)
    min_y = min(point[1] for point in data)
    max_y = max(point[1] for point in data)

    grid_x = ((max_x-min_x) / 3)+1
    grid_y = ((max_y-min_y) / 3)+1
    for i in data:
        column = int((i[0]-min_x) // grid_x)
        roll = int((i[1]-min_y) // grid_y)
        temp[roll][column] = [i[3] , i[2]]
        output[roll][column] = str(i[3])
    temp[0][1] = [""]
    for i in range(len(temp)):
        check.append([""] not in temp[i])
        
    # if True in check:
    #     print("Ihere")
    # else :
    #     print(output)

    # print(temp)
    # print(angle_degrees)
    # print(d_left)
    # print(d_right)
    if all(check) == True:
        print(555) 
        delta_x = max_x - min_x
        d_left = (temp[0][0][1] + temp[1][0][1] + temp[2][0][1]) / 3
        d_right = (temp[0][2][1] + temp[1][2][1] + temp[2][2][1]) / 3
        delta_depth = d_right - d_left
        angle_radians = math.atan2(delta_depth, delta_x)
        angle_degrees = round(math.degrees(angle_radians),4)
        return output
    else :
        print(1230129370)
        return 0
    
    
# find_table(data)
print(find_table(data))
# print(column,roll)
# print(min_x)
# print(min_y)
# print(grid_y)
# print(max_x)
# print(max_y)