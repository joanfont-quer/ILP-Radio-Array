from ska_ost_array_config import get_subarray_template

NAME = "SKA2subarrays"

def get_subarrays():
    assignments_1 = get_subarray_template("Mid_split2_1_AA4")
    assignments_2 = get_subarray_template("Mid_split2_2_AA4")
    return [assignments_1, assignments_2]


def get_ska_solution():
    subarrays = get_subarrays()
    ska_solution = {str(n): 0 for n in subarrays[0].array_config.names.data}
    ska_solution.update({str(n): 1 for n in subarrays[1].array_config.names.data})
    return ska_solution
