num_experts = 128
ep_size = 8

batch_size = 256
dp_size = 8
topk = 8

count = [0] * ep_size

for dp_rank in range(dp_size):
    for i in range(batch_size * topk):
        count[((dp_rank + i) % num_experts) // (num_experts // ep_size)] += 1

print(f"{batch_size = }, {dp_size = }, {count = }")
