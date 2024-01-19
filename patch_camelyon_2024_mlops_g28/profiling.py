import torch

import data.make_dataset as md
import predict_model as pm

MODEL_PDW = "models/test_model.pt"
PT_PDW = "data/testFiles/test_only_dataset.pt"

# code for profiling the functions within the make_datset.py file
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "patch_camelyon_2024_mlops_g28/profiling_logs/make_dataset_profiling"
    ),
) as make_data_prof:
    for i in range(1):
        data = md.h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_x.h5.gz", images=True)
        targets = md.h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_y.h5.gz", images=False)
        train_ds, test_ds, val_ds, test_images = md.main(data, targets)
        make_data_prof.step()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "patch_camelyon_2024_mlops_g28/profiling_logs/prediction_profiling"
    ),
) as predict_prof:
    for i in range(1):
        pm.main(MODEL_PDW, PT_PDW).shape
        predict_prof.step()
