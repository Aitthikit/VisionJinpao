import matplotlib.pyplot as plt
import numpy as np

# Define the points as a numpy array
points = np.array([
    [865, 228], [862, 231], [862, 254], [863, 255], [863, 261], [864, 262],
    [864, 305], [865, 306], [865, 330], [864, 331], [779, 331], [778, 332],
    [777, 332], [776, 333], [776, 352], [777, 352], [778, 353], [915, 353],
    [916, 354], [937, 354], [939, 356], [953, 356], [954, 357], [955, 357],
    [956, 358], [976, 358], [977, 357], [979, 357], [980, 356], [981, 356],
    [982, 355], [1011, 355], [1012, 354], [1013, 354], [1014, 353], [1017, 353],
    [1018, 352], [1030, 352], [1030, 333], [1026, 329], [998, 329], [997, 330],
    [963, 330], [962, 329], [960, 329], [959, 328], [958, 328], [957, 327],
    [956, 327], [953, 324], [953, 304], [954, 303], [954, 294], [955, 293],
    [955, 269], [956, 268], [956, 260], [957, 259], [957, 231], [937, 231],
    [936, 230], [935, 230], [934, 229], [932, 229], [931, 228]
])
print(points)
# Extract x and y coordinates
x = points[:, 0]
y = points[:, 1]

# Plot the points
plt.plot(x, y, marker='o', linestyle='-')
plt.fill(x, y, 'b', alpha=0.3)  # Fill the polygon with a semi-transparent color
plt.gca().invert_yaxis()  # Invert y-axis to match your data

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Polygon Plot')

# Show the plot
plt.grid(True)
plt.show()
