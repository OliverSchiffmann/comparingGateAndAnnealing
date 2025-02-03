# readMe:
# This solves the layout/tiling problem using grover's algorthim.
# Note this only works for an 8x8 grid

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
from qiskit_ibm_provider import least_busy
import json
import pandas as pd
import timeit


def save_to_json(
    filepath: str, filename, data
):  # exporting for alter analysis and plotting
    with open(filepath + filename, "w") as outfile:
        json.dump(data, outfile)


def compute():  # subroutine used as part of the oracle step in grover's algorithm
    # contraining the tiles to the eastern wall (x coords must = 111)
    qc.mct([x1_q[0], x1_q[1], x1_q[2]], c_q[0])
    qc.mct([x2_q[0], x2_q[1], x2_q[2]], c_q[1])

    qc.barrier()

    # Compares if y1_0 = y2_0
    qc.mct([y1_q[0], y2_q[0]], c_q[2])
    qc.x(y1_q[0])
    qc.x(y2_q[0])
    qc.mct([y1_q[0], y2_q[0]], c_q[2])
    qc.x(y1_q[0])
    qc.x(y2_q[0])

    qc.barrier()

    # Compares if y1_1 = y2_1
    qc.mct([y1_q[1], y2_q[1]], c_q[3])
    qc.x(y1_q[1])
    qc.x(y2_q[1])
    qc.mct([y1_q[1], y2_q[1]], c_q[3])
    qc.x(y1_q[1])
    qc.x(y2_q[1])

    qc.barrier()

    # Compares if y1_2 = y2_2
    qc.mct([y1_q[2], y2_q[2]], c_q[4])
    qc.x(y1_q[2])
    qc.x(y2_q[2])
    qc.mct([y1_q[2], y2_q[2]], c_q[4])
    qc.x(y1_q[2])
    qc.x(y2_q[2])

    qc.barrier()

    # constraining tiles to different positions (x coords will be the same so y coords must be different)
    qc.mct([c_q[2], c_q[3], c_q[4]], c_q[5])
    qc.x(c_q[5])

    return


def oracle():  # creating the oracle subroutine

    compute()
    qc.barrier()
    qc.mct([c_q[0], c_q[1], c_q[5]], v_q[0])
    qc.barrier()
    # uncompute
    compute()

    return


def initialise():  # creating linear superposition over qunatum register

    for q in x1_q:
        qc.h(q)
    for q in y1_q:
        qc.h(q)
    for q in x2_q:
        qc.h(q)
    for q in y2_q:
        qc.h(q)
    qc.initialize([1, -1] / np.sqrt(2), v_q)

    return


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits - 1)
    qc.mct(list(range(nqubits - 1)), nqubits - 1)  # multi-controlled-toffoli
    qc.h(nqubits - 1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s


def sim(
    ibm_token, device, required_qubits, num_shots, MAX_EXECUTION_TIME, num_oracles
):  # submits the circuit as a job and returns the results from IBM
    if test_mode != 1:  # just for testing with a simple circuit
        qc.measure([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], cbits)

    QiskitRuntimeService.save_account(
        channel="ibm_quantum", token=ibm_token, overwrite=True
    )
    service = QiskitRuntimeService()
    options = Options(
        optimization_level=1,
        max_execution_time=MAX_EXECUTION_TIME,
        environment={
            "job_tags": [
                "shots="
                + str(num_shots)
                + ", oracles="
                + str(num_oracles)
                + "_device="
                + str(device)
            ]
        },
    )

    if device == "fake":
        with Session(service=service, backend="ibmq_qasm_simulator") as session:
            sampler = Sampler(session=session, options=options)
            job = sampler.run(circuits=qc, shots=num_shots)
            print(job.job_id(), job.status(), job.backend().name)
            result = job.result()
            print(job.usage_estimation)
            print(result)

        print("session timeout")

    if device == "real":
        small_devices = service.backends(
            min_num_qubits=required_qubits, simulator=False, operational=True
        )
        backend = least_busy(
            small_devices
        )  # chooses the least busy device (shortest queue)
        # backend = "ibm_brisbane" # can specify a device using this

        with Session(service=service, backend=backend, max_time=601) as session:
            sampler = Sampler(session=session, options=options)
            job = sampler.run(circuits=qc, shots=num_shots)
            print(job.job_id(), job.status(), job.backend().name)
            result = job.result()
            print(job.usage_estimation)
            print(result)

        print("session timeout")

    return result, job.job_id()


def convert_int_key_to_binary(
    results_dict,
):  # used for processing the raw results (based on register positions) from IBM into a binary string
    binary_results = {}
    for dictionary in results_dict:
        for key, value in dictionary.items():
            binary_key = format(
                key, "012b"
            )  # Convert integer to 12-digit binary string
            reversed_binary_key = binary_key[::-1]  # Reverse the binary string
            binary_results[reversed_binary_key] = (
                value  # Store reversed binary string as key in a new dictionary
            )
    return binary_results


def quasi_prob_to_frequency(
    results, num_shots
):  # turning probability into frequnecy of placement for tile results plotting
    updated_results = {
        key: round(value * num_shots) for key, value in results.items()
    }  # needs rounding for plotting later.
    return updated_results


def dict_to_array(results_d_2_a):
    array_result = []
    for key, value in results_d_2_a.items():
        array_result.append([key[:6], key[6:12], value])
    return array_result


def results_to_excel(array, folder, file_name):
    df = pd.DataFrame(array)
    file_path = f"{folder}{file_name}"
    df.to_excel(file_path, index=False, header=False)


def same_pos_check(data):  # basic analysis for trouble shooting
    count = 0
    for row in data:
        if row[0] == row[1]:
            count += 1
    print(f"Number of rows with the first two items identical: {count}")


def extract_quasi_dists(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    quasi_dists = data["quasi_dists"][0]

    return quasi_dists


test_mode = 0  # 1 if you want to run the 2 qubit test example
num_oracles = 9  # sqrt(N/M) is ideal
num_shots = 40  # number of times the circuit is repeated
device = "fake"  # real if you want to use the least busy quantum device with enough qubits, fake if you want to use the quantum statevector simulator
my_token = "IBM token"
filename = (
    "Grovers_"
    + device
    + "_results_shots"
    + str(num_shots)
    + "_oracles"
    + str(num_oracles)
    + ".xlsx"
)  # choosing the file name
outputs_filepath = "/output/file/path/name"  # saving it in the design 24 file path


if (
    __name__ == "__main__"
):  # only runs when the script is executed directly, not when it is imported as a module, this part creates the circuit
    if test_mode == 1:
        qc = QuantumCircuit(2)

        qc.h(0)
        qc.h(1)

        qc.measure_all()
        print(qc)
    else:
        x1_q = QuantumRegister(3, "x1")
        y1_q = QuantumRegister(3, "y1")
        x2_q = QuantumRegister(3, "x2")
        y2_q = QuantumRegister(3, "y2")

        c_q = QuantumRegister(6, "c")
        v_q = QuantumRegister(1, "v")

        cbits = ClassicalRegister(12, "cbits")

        qc = QuantumCircuit(x1_q, y1_q, x2_q, y2_q, c_q, v_q, cbits)

        initialise()

        for i in range(0, num_oracles):  # 31
            qc.barrier()
            oracle()
            qc.barrier()
            qc.append(diffuser(12), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

print(qc)

# %%
start = timeit.default_timer()  # to get timing data

# sim(token, device, min qubits, shots, max execution time in seconds, number of oracle repetitions)
result, job_id = sim(
    my_token, device, qc.num_qubits, num_shots, 600, num_oracles
)  # this line runs the program.

print("The total time for results in seconds is:", timeit.default_timer() - start)

# %%
process_results = "yes"  # result processing is seperate as when you submit to a real device it will be placed in a queue for a long time
job_id = "jobID"  # to find this you can copy it from the ibm job manager. But you will also need to download the results and store them in the file path specified by outputs_filepath
jobs_filepath = "/filepath/for/downloaded/job/results"


def reverse_dictionary_keys(original_dict):
    reversed_dict = {key[::-1]: value for key, value in original_dict.items()}
    return reversed_dict


if process_results == "yes":

    file_path = (
        jobs_filepath + "job-" + str(job_id) + "/" + str(job_id) + "-result.txt"
    )  # Replace 'your_file.txt' with the path to your file
    backwards_result_as_binary = extract_quasi_dists(file_path)
    result_as_binary = reverse_dictionary_keys(backwards_result_as_binary)
    result_binary_frequency = quasi_prob_to_frequency(result_as_binary, num_shots)
    result_array = dict_to_array(result_binary_frequency)
    same_pos_check(result_array)
    results_to_excel(result_array, outputs_filepath, filename)
