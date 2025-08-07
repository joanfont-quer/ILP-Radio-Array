from ska_ost_array_config.array_config import MidSubArray
from utils import describe_subarray


template = MidSubArray(subarray_type="custom", custom_stations="M000,M001,M002")
describe_subarray(template, 5, 2, 1, 1)
