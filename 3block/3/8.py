def count_parameters_conv(in_channels: int, 
                         out_channels: int, 
                         kernel_size: int, 
                         bias: bool) -> int:

    weight = in_channels*out_channels*(kernel_size**2)
    return weight+out_channels if bias else weight


print(count_parameters_conv(2,2,2,2))

