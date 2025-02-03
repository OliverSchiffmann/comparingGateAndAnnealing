# READ ME:
# This script is used for plotting the results from the BQM and Grover's approach to the tiling problem.
# Results are imported in a standard format from an excel sheet.

# %%Importing the necessary packages
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

# %The functions used for this script are:
# Converting the data that is stored in the respective excel sheet into a numpy array so that it can be operated on.
# SumColumn is just a small fucntion i use to check that i havent lost any data in the import. Frequnecies should add up to the number of runs the sampler made.
# Plot 3d histogram is the plotting function. it takes in data in the form of a matrix the same dimensions as the tile grid. But in each position is the frequency with which tile 1 or 2 is placed there (depending on which set of data youre plotting).
# results to 2d array takes the placement of 1 tile and its frequencies, as it comes out of the excel sheet, and arranges it into the 8x8 matrix format where each value is the frequnecy of placements in that position.


def convert(
    filename, filePath
):  # this opens an excel sheets and converts it into a numpy array
    workbook = openpyxl.load_workbook(filePath + filename)

    # Choose a specific sheet
    sheet = workbook.active  # or specify the sheet name like: workbook['Sheet1']

    # Convert sheet data to a NumPy array
    results = np.array([list(row) for row in sheet.iter_rows(values_only=True)])

    return results


def extract_grid_width(results_local):
    first_element = results_local[
        0, 0
    ]  # Assuming 'array' is a NumPy array or a similar structure
    binary_length = len(str(first_element))
    half_length = binary_length // 2
    largest_integer = 2**half_length
    return largest_integer


def sumColumn(
    array, col_num
):  # this summs up the values of a specified column in a specified array as integers
    column_sum = np.sum(array[:, col_num].astype(int))

    return column_sum


def plot_3d_histogram(valueArray, plot_title, bar_colour, view):
    # Convert the input array to a numpy array
    data_array = np.array(valueArray)

    data_array += (
        1  # adding 1 to all positions so that there arent those ugly grey squares
    )

    # Create a figure for plotting the data as a 3D histogram
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if view == "front":
        ax.view_init(elev=45, azim=240)
    elif view == "back":
        ax.view_init(elev=45, azim=60)

    # Create an X-Y mesh of the same dimension as the 2D data
    x_data, y_data = np.meshgrid(
        np.arange(data_array.shape[1]), np.arange(data_array.shape[0])
    )

    # Flatten out the arrays for use in ax.bar3d
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()

    # Plot the 3D bars
    ax.bar3d(
        x_data,
        y_data,
        np.zeros(len(z_data)),
        1,
        1,
        z_data,
        shade=True,
        color=bar_colour,
    )

    # Add axis labels
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)

    plt.title(plot_title)

    # Save figures
    fig.savefig("grid_figure.png", dpi=600)

    # Display the plot
    plt.show()


def results_to_2d_array(data, tileNumber, gridWidth, valueColumn):
    # Create a 2D array to store the frequencies at each position
    valueArray = np.zeros((gridWidth, gridWidth))
    coordLength = len(bin(abs(gridWidth - 1))) - 2

    # Iterate through your data and populate the frequency array
    for row in data:
        binary_x = int(row[tileNumber][:coordLength], 2)
        binary_y = int(row[tileNumber][coordLength:], 2)
        values = float(row[valueColumn])
        valueArray[binary_y, binary_x] += values

    return valueArray


def noise_level_eastern_wall(data_array):
    invalid_solutions_count = 0
    total = sum(int(row[-1]) for row in data_array)

    for row in data_array:
        if "0" in row[0][:3] or "0" in row[1][:3]:
            invalid_solutions_count += int(row[-1])

    percentage = (invalid_solutions_count / total) * 100 if total > 0 else 0
    return percentage


def noise_level_same_position(data_array):
    invalid_solutions_count = 0
    total = sum(int(row[-1]) for row in data_array)

    for row in data_array:
        if row[0] == row[1]:
            invalid_solutions_count += int(row[-1])

    percentage = (invalid_solutions_count / total) * 100 if total > 0 else 0
    return percentage


def measure_uniformity(data):
    """
    Measures the uniformity of a 1D array containing frequency data.

    Args:
      data: A 1D NumPy array of frequencies.

    Returns:
      A tuple containing:
        - uniformity_score: A float between 0 and 1, where 1 indicates perfect uniformity.
        - explanation: A string explaining the chosen metric and its interpretation.

    Raises:
      ValueError: If the data array is not 1D.
    """

    if data.ndim != 1:
        raise ValueError("Data must be a 1D array.")

    uniformity_score = np.std(data) / np.mean(data)
    explanation = "Coefficient of variation (standard deviation / mean), where a lower value indicates higher uniformity."

    return uniformity_score, explanation


# %%Here i am jsut selecting the excel sheet we want to look at. Note that it must be stored in the DESIGN 24/results file path
Filename = "BQM_fake_results_shots1000_east_wall_weight3_penalty0.5.xlsx"
FilePath = "/Users/kv18799/Library/CloudStorage/OneDrive-UniversityofBristol/Documents/PhD/Year 1/QC 4 Eng/DESIGN 24/Results/Final_4_dcc_old/"

# %%Here i am running all the functions to plot the histogram, as well as printing the results along the way to help spot bugs.
Results = convert(
    Filename, FilePath
)  # extracting the results from the specified excel file

GridWidth = extract_grid_width(Results)

print("The width of the grid was:", GridWidth)

print(
    "A brief summary of the results stored in the Excel file looks like:\n", Results
)  # printing the first and last few results obtained from the excel file

print(
    "Number of times the solver was run was:\n", sumColumn(Results, 2)
)  # printing the number of placements detected in the excel file. Should match the numbers of shots specified for the sampler

frequency_array_0 = results_to_2d_array(
    Results, 0, GridWidth, 2
)  # creating the frequency array for plotting the positions of tile 1
frequency_array_1 = results_to_2d_array(
    Results, 1, GridWidth, 2
)  # creating the frequency array for plotting the positions of tile 2
print(
    "The frequency of tile placement for tile 1 in each position is shown below in a matrix:\n",
    frequency_array_0,
)  # Printing the raw results before formatting as a histogram for bug spotting
print(
    "The frequency of tile placement for tile 2 in each position is shown below in a matrix:\n",
    frequency_array_1,
)  # Printing the raw results before formatting as a histogram for bug spotting

plot_3d_histogram(
    frequency_array_0, "", "pink", "front"
)  # Plotting the histogram for 1 tile, and also specifying the title of that plot as well as its colour and view direction
# circuit green #4FA582
# circuit gold #F8D688


plot_3d_histogram(
    frequency_array_1, "", "orange", "front"
)  # Options for view are only 'front' or 'back'


# %%analysis of noise in results
eastern_wall_noise = noise_level_eastern_wall(Results)
print(
    "The percentage of solutions that have at least 1 tile not on the eastern wall is "
    + str(eastern_wall_noise)
    + "%"
)

same_pos_noise = noise_level_same_position(Results)
print(
    "The percentage of solutions that have tiles in the same position is "
    + str(same_pos_noise)
    + "%"
)

print("The total noise is " + str(same_pos_noise + eastern_wall_noise) + "%")

# %% Energy Landscape plotting
EnergyAnalysis = False

if EnergyAnalysis:
    EnergyFilename = (
        "BQM_energies_grid8_fake_results_shots5_east_wall_weight3_penalty0.5.xlsx"
    )

    EnergyResults = convert(EnergyFilename, FilePath)
    EnergyArray = np.abs(results_to_2d_array(EnergyResults, 0, GridWidth, 1))

    print(
        "The absolute values for the energy of each tile 1 position are:\n", EnergyArray
    )

    # plot_3d_histogram(EnergyArray, "Absolute of Energies for tile 2 = [1,1]", 'orange', 'front')
    plot_3d_histogram(EnergyArray, "", "orange", "front")

# %% Uniform distribution analysis

score, explanation = measure_uniformity(frequency_array_0[:, -1])
print(f"Uniformity score for Tile 1's placement: {score:.4f}")

score, explanation = measure_uniformity(frequency_array_1[:, -1])
print(f"Uniformity score for Tile 2's placement: {score:.4f}")
print(f"Explanation: {explanation}")
