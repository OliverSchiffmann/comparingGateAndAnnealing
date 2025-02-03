# READ ME:
# A summary of this script is as follows. Overall this is a BQM approach to solving the tiling problem using a simulated D-wave QPU using DCC constraints.
# This script also has an iterating function. This allows it to be run mulitple times. collating_results.py can then be used to obtain data about averaged results


from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel, ExactSolver
from neal import (
    SimulatedAnnealingSampler,
)  # this will need updating as it will be removed in dwave-ocean-sdk v8.0.0
import pandas as pd
import json
import timeit


def export_to_excel(
    result_array, filename, filepath
):  # exporting the data so that it can be used for alter analysis and plotting
    # Create a DataFrame from the result_array
    df = pd.DataFrame(result_array, columns=["String", "Middle", "Frequency"])

    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(filepath + filename, engine="xlsxwriter")

    # Convert the DataFrame to an XlsxWriter Excel object without column headings
    df.to_excel(writer, sheet_name="Sheet1", index=False, header=False)

    # Close the Pandas Excel writer and output the Excel file
    writer.save()


def save_to_json(
    filepath: str, filename, data
):  # needed for saving the timing data from QPU achieved results
    with open(filepath + filename, "w") as outfile:
        json.dump(data, outfile)


def required_variables(
    width,
):  # Working out the number of binary variables required for the given grid width
    num_req_variables = (len(bin(abs(width - 1))) - 2) * 4
    return num_req_variables


def create_grid(
    num_req_variables,
):  # creating an empty BQM with the right number of variables
    bqm = BinaryQuadraticModel("BINARY")

    for i in range(1, num_req_variables + 1):  # creating all the binary variables
        xi = f"x{i}"
        bqm.add_variable(xi, 0)

    return bqm


def east_wall_constraint(
    num_req_variables, bqm, penalty
):  # creating the constraint that tiles must be placed on the eastern wall, done by creating a penalty function

    coord_len = int(num_req_variables / 4)

    for i in range(1, num_req_variables + 1):  # iterating over all the variables
        xi = f"x{i}"

        if (
            i <= coord_len or (coord_len * 2) + 1 <= i <= coord_len * 3
        ):  # rewarding x coordinates that contain 1s (are closer to east wall)
            bqm.add_variable(xi, -penalty)
        else:
            bqm.add_variable(xi)
    return bqm


def no_overlap_constraint(
    num_req_variables, bqm, penalty
):  # creating the constraint that both tiles cannot be palced in the same location
    bqm.add_variable(
        "z", -penalty * 6
    )  # Adding the ancilla variable, large negative bias to encourage z = 1

    var_per_tile = int(num_req_variables / 2)

    # Loop to add terms for each pair
    for i in range(1, int(var_per_tile) + 1):
        x_sq1 = f"x{i}"
        x_sq2 = f"x{i + var_per_tile}"

        # Add XNOR constraint for this pair
        bqm.add_interaction(x_sq1, x_sq2, 2 * penalty)
        bqm.add_interaction(x_sq1, "z", -2 * penalty)
        bqm.add_interaction(x_sq2, "z", -2 * penalty)
        bqm.add_variable(x_sq1, penalty)
        bqm.add_variable(x_sq2, penalty)
        bqm.add_variable("z", penalty)
    return bqm


def extract_tiles(
    sampleset_local, num_req_variables
):  # extracting just the necessary info from the data returned by D-wave
    coord_string_len = int(num_req_variables / 2)
    response_local = []
    for sample, energy, num_occurrences in sampleset_local.data(
        ["sample", "energy", "num_occurrences"]
    ):
        tile_1 = "".join(str(sample[f"x{v}"]) for v in range(1, coord_string_len + 1))
        tile_2 = "".join(
            str(sample[f"x{v}"])
            for v in range(coord_string_len + 1, num_req_variables + 1)
        )
        combined = {
            "tile 1": tile_1,
            "tile 2": tile_2,
            "energy": energy,
            "num_occurences": num_occurrences,
            "overlap": tile_1 == tile_2,
        }
        if combined["overlap"]:
            print(combined)
        response_local.append(combined)
    return response_local


def extract_energies(
    landscape, exactSampleset, numReqVar
):  # extracting data about the energy (performance) of each solution from the previousl;y extracted dataset
    if landscape:
        coordStringLen = int(numReqVar / 2)
        exactResponse = []
        for sample, energy in exactSampleset.data(["sample", "energy"]):
            if (
                sample["z"] == 1
            ):  # unsure what the implication of looking at z == 1 vs z == 0 would be
                tile_1 = "".join(
                    str(sample[f"x{v}"]) for v in range(1, coordStringLen + 1)
                )
                tile_2 = "".join(
                    str(sample[f"x{v}"])
                    for v in range(coordStringLen + 1, numReqVar + 1)
                )
                combined = {"tile 1": tile_1, "tile 2": tile_2, "energy": energy}
                exactResponse.append(combined)
        return exactResponse


def slice_energy_response(
    landscape, exactResponse, tile2Pos
):  # used for checking the energy of a particular solution in troubleshooting
    if landscape:
        slicedExactResponse = []
        for sample in exactResponse:
            if sample["tile 2"] == tile2Pos:
                slicedExactResponse.append(sample)
        return slicedExactResponse


def generate_filename(
    jobNumber, device_type, grid_size, num_run, e_weight, overlap_weight, landscape
):  # using unique file names generated based on parameters for later analysis and plotting

    name = (
        str(jobNumber)
        + "_BQM_grid"
        + str(grid_size)
        + "_"
        + str(device_type)
        + "_results_shots"
        + str(num_run)
        + "_east_wall_weight"
        + str(e_weight)
        + "_penalty"
        + str(overlap_weight)
        + ".xlsx"
    )
    energyName = "N/A"
    if landscape:
        energyName = (
            "BQM_energies_grid"
            + str(grid_size)
            + "_"
            + str(device_type)
            + "_results_shots"
            + str(num_run)
            + "_east_wall_weight"
            + str(e_weight)
            + "_penalty"
            + str(overlap_weight)
            + ".xlsx"
        )

    return name, energyName


def save_problem_timings(
    jobNumber,
    device_type,
    gridSize,
    save_location,
    num_run,
    e_weight,
    overlap_weight,
    sampleset,
    start_time,
):  # using unique file names generated based on parameters for later analysis and plotting
    if device_type == "real":
        save_to_json(
            save_location,
            str(jobNumber)
            + "_BQM_timing_grid"
            + str(gridSize)
            + "_"
            + str(num_run)
            + "_"
            + str(e_weight)
            + "_"
            + str(overlap_weight)
            + ".json",
            sampleset.info["timing"],
        )
    # Saving the timing data to a josn folder so that it can be read without resubmitting the job
    print(
        "The total time for results in seconds is:", timeit.default_timer() - start_time
    )


def enter_the_realm_of_uncertainty(
    jobNumber,
    deviceType,
    landscape,
    samples,
    gridSize,
    eastWallWeight,
    noOverlapWeight,
    outputFilepath,
):  # the script that submitts the job and returns the results
    numReqVar = required_variables(gridSize)

    bqm = create_grid(numReqVar)
    bqm = east_wall_constraint(numReqVar, bqm, eastWallWeight)
    bqm = no_overlap_constraint(numReqVar, bqm, noOverlapWeight)

    if deviceType == "fake":
        sampler = SimulatedAnnealingSampler()
    elif deviceType == "real":
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        print(
            "Device should be real if using dedicated solver or fake if using simulated annealing solver."
        )

    start = timeit.default_timer()
    sampleset = sampler.sample(bqm, num_reads=samples)
    save_problem_timings(
        jobNumber,
        deviceType,
        gridSize,
        outputFilepath,
        samples,
        eastWallWeight,
        noOverlapWeight,
        sampleset,
        start,
    )

    if landscape:
        exactSampleset = ExactSolver().sample(bqm)
        return sampleset, exactSampleset

    return sampleset, None


# %%Setting up the problem, creating the varibles and imposing the constraints
# Global Variables
Device = "fake"  # fake for a simulator real for QPU
Shots = 5  # number of repetitions of the annealing algorithm per job
Repetitions = 1  # number of times the job is submitted, needed to get data about start to end timings
GridSize = 8  # options are 8, 16, 32, 64
EastWallWeight = 3  # weighting of the constraint
NoOverlapWeight = 0.5  # weighting of the constraint
OutputsFilepath = "/Users/kv18799/Library/CloudStorage/OneDrive-UniversityofBristol/Documents/PhD/Year 1/QC 4 Eng/DESIGN 24/Results/Repeated_jrnl_results/"
Landscape = False  # True if you want to find the energy of every possible solution, False if you dont.


for JobNumber in range(0, Repetitions):

    Sampleset, ExactSampleset = enter_the_realm_of_uncertainty(
        JobNumber,
        Device,
        Landscape,
        Shots,
        GridSize,
        EastWallWeight,
        NoOverlapWeight,
        OutputsFilepath,
    )

    response = extract_tiles(
        Sampleset, required_variables(GridSize)
    )  # Here we are getting the results out of Sampleset/ExactSampleset that we are interested in

    # Below are relevant if you are investigating the energy of every solution using ExactSolver
    ExactResponse = extract_energies(
        Landscape, ExactSampleset, required_variables(GridSize)
    )
    SlicedExactResponse = slice_energy_response(
        Landscape, ExactResponse, "001000"
    )  #'xxxxxx' is the selection of energies we want to look at for when tile 2 is in position xxxxxx

    response_pd = pd.DataFrame(
        response
    )  # Here we are exporting the results we are interested in for plotting with another script
    SlicedExactResponsePd = pd.DataFrame(SlicedExactResponse)

    file_name, EnergyFileName = generate_filename(
        JobNumber, Device, GridSize, Shots, EastWallWeight, NoOverlapWeight, Landscape
    )

    response_pd.to_excel(
        OutputsFilepath + file_name,
        index=False,
        header=False,
        columns=["tile 1", "tile 2", "num_occurences"],
    )

    if Landscape:
        SlicedExactResponsePd.to_excel(
            OutputsFilepath + EnergyFileName,
            index=False,
            header=False,
            columns=["tile 1", "energy"],
        )
