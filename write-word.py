import numpy as np 
import matplotlib.pyplot as plt


def generate_p(num_points=100, width=1):
    vertical_line = distribute_points_around_line(np.array([1, 1]), np.array([1, 9]), num_points // 2, width)
    semi_circle = distribute_points_around_circle(np.array([1, 7]), 2, -180, 0, num_points // 2, width)
    return np.hstack((vertical_line, semi_circle))

def generate_u(num_points=100, width=1):
    left_vertical = distribute_points_around_line(np.array([4, 1]), np.array([4, 5]), num_points // 3, width)
    bottom_curve = distribute_points_around_circle(np.array([5, 3]), 1, 180, 0, num_points // 3, width)
    right_vertical = distribute_points_around_line(np.array([6, 5]), np.array([6, 1]), num_points // 3, width)
    return np.hstack((left_vertical, bottom_curve, right_vertical))


def generate_a(num_points=100, width=1):
    left_slant = distribute_points_around_line(np.array([4.5, 1]), np.array([5.5, 9]), num_points // 3, width)
    right_slant = distribute_points_around_line(np.array([5.5, 9]), np.array([6.5, 1]), num_points // 3, width)
    cross_bar = distribute_points_around_line(np.array([4.75, 5]), np.array([6.25, 5]), num_points // 3, width)
    return np.hstack((left_slant, right_slant, cross_bar))


def generate_l(num_points=100, width=1):
    vertical_line = distribute_points_around_line(np.array([7, 9]), np.array([7, 1]), num_points // 2, width)
    horizontal_line = distribute_points_around_line(np.array([7, 1]), np.array([8.5, 1]), num_points // 2, width)
    return np.hstack((vertical_line, horizontal_line))




def distribute_points_around_line(start, end, num_points, width=1):
    """Distribute points around a line to simulate stroke width."""
    x_values = np.linspace(start[0], end[0], num_points)
    y_values = np.linspace(start[1], end[1], num_points)
    # Adding randomness to simulate the stroke width
    x_values += np.random.uniform(-width/2, width/2, num_points)
    y_values += np.random.uniform(-width/2, width/2, num_points)
    return np.vstack((x_values, y_values))

def distribute_points_around_circle(center, radius, start_angle, end_angle, num_points, width=1):
    """Distribute points around a part of a circle to simulate stroke width."""
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num_points)
    x_values = center[0] + radius * np.cos(angles) + np.random.uniform(-width/2, width/2, num_points)
    y_values = center[1] + radius * np.sin(angles) + np.random.uniform(-width/2, width/2, num_points)
    return np.vstack((x_values, y_values))

# Adjusted number of points for equal distribution
num_points_per_letter = 400
width = 0.1  # Adjusting for a finer stroke width

# Generate points for each letter
n_points = np.hstack([
    distribute_points_around_line(np.array([1, 9]), np.array([1, 1]), num_points_per_letter // 3, width),
    distribute_points_around_line(np.array([1, 9]), np.array([3, 1]), num_points_per_letter // 3, width),
    distribute_points_around_line(np.array([3, 1]), np.array([3, 9]), num_points_per_letter // 3, width)
])
i_points = distribute_points_around_line(np.array([4.5, 1]), np.array([4.5, 9]), num_points_per_letter, width)
c_points = distribute_points_around_circle(np.array([7.5, 5]), 2, 90, 270, num_points_per_letter, width) * np.array([1, 2])[..., None] - np.array([0, 5])[..., None]
o_points = distribute_points_around_circle(np.array([10.5, 5]), 2, 0, 360, num_points_per_letter, width)* np.array([1, 2])[..., None] - np.array([0, 5])[..., None]



# Combine all points correctly
all_points = np.hstack((n_points, i_points, c_points, o_points))

plt.figure(figsize=(10, 5))
plt.scatter(all_points[0], all_points[1], s=1)
plt.title('Distribution Resembling the Word "NICO"')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.show() 





















def generate_m(num_points=100, width=1):
    left_vertical = distribute_points_around_line(np.array([1, 1]), np.array([1, 9]), num_points // 4, width)
    middle_to_top = distribute_points_around_line(np.array([1, 9]), np.array([2.5, 5]), num_points // 4, width)
    middle_to_bottom = distribute_points_around_line(np.array([2.5, 5]), np.array([4, 9]), num_points // 4, width)
    right_vertical = distribute_points_around_line(np.array([4, 9]), np.array([4, 1]), num_points // 4, width)
    return np.hstack((left_vertical, middle_to_top, middle_to_bottom, right_vertical))

def generate_a(num_points=100, width=1):
    left_slant = distribute_points_around_line(np.array([4.5, 1]), np.array([5.5, 9]), num_points // 3, width)
    right_slant = distribute_points_around_line(np.array([5.5, 9]), np.array([6.5, 1]), num_points // 3, width)
    cross_bar = distribute_points_around_line(np.array([4.75, 5]), np.array([6.25, 5]), num_points // 3, width)
    return np.hstack((left_slant, right_slant, cross_bar))

def generate_r(num_points=100, width=1):
    vertical_line = distribute_points_around_line(np.array([7, 1]), np.array([7, 9]), num_points // 3, width)
    semi_circle = distribute_points_around_circle(np.array([7, 7.5]), 1.5, -90, 90, num_points // 3, width)
    leg = distribute_points_around_line(np.array([7, 6]), np.array([8.5, 1]), num_points // 3, width)
    return np.hstack((vertical_line, semi_circle, leg))

# Adjusting positions for "C" and "O" for the word "MARCO"
def generate_c_marco(num_points=100, width=1):
    return distribute_points_around_circle(np.array([11.5, 5]), 2, 90, 270, num_points, width)* np.array([1, 2])[..., None] - np.array([0, 5])[..., None]

def generate_o_marco(num_points=100, width=1):
    return distribute_points_around_circle(np.array([14, 5]), 2, 0, 360, num_points, width)* np.array([1, 2])[..., None] - np.array([0, 5])[..., None]

# Generate points for each letter with adjusted parameters
m_points = generate_m(num_points_per_letter, width)
a_points = generate_a(num_points_per_letter, width)
r_points = generate_r(num_points_per_letter, width)
c_points_marco = generate_c_marco(num_points_per_letter, width)
o_points_marco = generate_o_marco(num_points_per_letter, width)

# Combine all points
all_points_marco = np.hstack((m_points, a_points, r_points, c_points_marco, o_points_marco))

plt.figure(figsize=(12, 5))
plt.scatter(all_points_marco[0], all_points_marco[1], s=1)
plt.title('Distribution Resembling the Word "MARCO"')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





assert False 
# Reusing the A generation function from "MARCO"
a_points_paul = generate_a(num_points_per_letter, width)

# Generate points for new letters "P," "U," and "L"
p_points = generate_p(num_points_per_letter, width)
u_points = generate_u(num_points_per_letter, width)
l_points = generate_l(num_points_per_letter, width)

# Combine all points
all_points_paul = np.hstack((p_points, a_points_paul, u_points, l_points))

plt.figure(figsize=(10, 5))
plt.scatter(all_points_paul[0], all_points_paul[1], s=1)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title('Distribution Resembling the Word "PAUL"')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()