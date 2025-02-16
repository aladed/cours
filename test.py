import torch
import time

N = 30_000  
density = 0.0001 

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1) 


indices = torch.randint(0, N, (2, int(N * N * density))) 
values = torch.randn(int(N * N * density)) 

sparse_matrix_cpu = torch.sparse_coo_tensor(indices, values, (N, N), device=device_cpu)
sparse_matrix_gpu = sparse_matrix_cpu.to(device_gpu)

dense_matrix_cpu = torch.randn(N, N, device=device_cpu)
dense_matrix_gpu = dense_matrix_cpu.to(device_gpu)


start = time.time()
result_cpu = torch.sparse.mm(sparse_matrix_cpu, dense_matrix_cpu)
end = time.time()
print(f"PyTorch (CPU, 1 поток): {end - start:.3f} сек")


start = time.time()
result_gpu = torch.sparse.mm(sparse_matrix_gpu, dense_matrix_gpu)
torch.cuda.synchronize()
end = time.time()
print(f"PyTorch (GPU RTX 3080): {end - start:.3f} сек")

