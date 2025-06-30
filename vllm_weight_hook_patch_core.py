# vllm_weight_hook_patch_core.py
'''
Usage: 
Use install_vllm_hook_patch.sh to install the VLLM patch.
Use uninstall_vllm_hook_patch.sh to remove the patch.
Or manually:
Put vllm_patch_loader.py & vllm_weight_hook_patch_core.py & vllm_injector.pth into the python site-packages directory of the target environment.
Use this command to find the site-packages directory:
python -c "import site; print(site.getsitepackages()[0])"
'''

import sys
import os
import fcntl
# import runpy
import json
import time
import torch
from typing import List, Tuple, Union, Optional

import vllm.distributed.parallel_state as parallel_state_module

print(f"[VLLM_PATCH] Patch Module loaded in process: {os.getpid()}")
# ===================================================================
# All patching code for model runners to handle weight metadata saving
def _patched_acquire_weight_lock(self, timeout=10):
    """acquire weight metadata saving file lock"""
    os.makedirs("vllm_weights_metadata", exist_ok=True)
    lock_file = os.path.join("vllm_weights_metadata", f"weight_saving_{parallel_state_module.get_tensor_model_parallel_rank()}.lock")

    try:
        self._lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
        start_time = time.time()

        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # logger.info(f"Acquired weight saving lock for GPU {self.gpu_id}")
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    # logger.error(f"Failed to acquire weight lock within {timeout} seconds")
                    os.close(self._lock_fd)
                    return False
                time.sleep(0.1)
    except Exception as e:
        # logger.error(f"Error acquiring weight lock: {e}")
        return False

def _patched_release_weight_lock(self):
    """release weight metadata saving file lock"""
    if hasattr(self, '_lock_fd'):
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            # delete lock file
            lock_file = os.path.join("vllm_weights_metadata", f"weight_saving_{parallel_state_module.get_tensor_model_parallel_rank()}.lock")
            if os.path.exists(lock_file):
                os.remove(lock_file)
            # logger.info(f"Released weight saving lock for GPU {self.gpu_id}")
        # except Exception as e:
            # logger.warning(f"Error releasing weight lock: {e}")
        finally:
            delattr(self, '_lock_fd')

# Weights_hook function 
def _patched_register_weight_hooks(self):
    # self.weight_infos = {}  # Save weight metadatas
    self._clear_old_weight_data()

    def tensor_hook(tensor: torch.Tensor, name: str):
        if tensor.is_cuda:
            self.weight_infos[name] = {
                "ptr": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                # "actual_size": tensor.storage().size() * tensor.element_size(),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape)
            }

    if not self._acquire_weight_lock():
        raise RuntimeError("Failed to acquire weight metadata update lock")

    # Register hooks to capture the initial state of model weights
    for name, param in self.model.named_parameters():
        tensor_hook(param, name)  # Capture parameter weights
    self._save_weight_meta()  # Save weight metadata to a local file
    self.total_weight_dict = self._calculate_device_weight_sizes(unit="GB")
    self._save_total_weight_meta()
    # self._merge_weights()  # Merge weights based on pointer continuity
    # self._save_merged_weight_meta()  # Save merged weight metadata to a local file
    self._release_weight_lock()

# Save the model weight metadata to a JSON file
def _patched_save_weight_meta(self):
    os.makedirs("vllm_weights_metadata", exist_ok=True)
    meta_path = os.path.join("vllm_weights_metadata", f"weights_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.weight_infos, f, indent=2)
        # logger.info(f"Save weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save weight metadata to {meta_path}: {e}")
        raise

def _patched_save_total_weight_meta(self):
    os.makedirs("vllm_weights_metadata", exist_ok=True)
    meta_path = os.path.join("vllm_weights_metadata", f"total_weight_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.total_weight_dict, f, indent=2)
        # logger.info(f"Save total weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save total weight metadata to {meta_path}: {e}")
        raise

def _patched_calculate_device_weight_sizes(self, unit: str = "bytes") -> dict:
    """Calculate the total size of weights per device in self.weight_infos.
    
    Args:
        unit (str): The unit to return the size in. 
                    Options: "bytes", "KB", "MB", "GB".
    
    Returns:
        dict: {device: total_size} where total_size is in the specified unit.
    """
    device_sizes = {}  # {device: total_size_in_bytes}

    # 遍历所有 weight_infos，按 device 累加 size
    for info in self.weight_infos.values():
        device = info["device"]
        size = info["size"]
        if device in device_sizes:
            device_sizes[device] += size
        else:
            device_sizes[device] = size

    # 单位转换
    unit = unit.upper()
    if unit == "KB":
        return {device: size / 1024 for device, size in device_sizes.items()}
    elif unit == "MB":
        return {device: size / (1024 ** 2) for device, size in device_sizes.items()}
    elif unit == "GB":
        return {device: size / (1024 ** 3) for device, size in device_sizes.items()}
    else:  # Default to bytes
        return device_sizes
    
def _patched_clear_old_weight_data(self):
    """
    Clear old weight information and metadata files
    """
    # Clear in-memory data
    if hasattr(self, 'weight_infos'):
        self.weight_infos.clear()
    else:
        self.weight_infos = {}

    if hasattr(self, 'total_weight_dict'):
        self.total_weight_dict.clear()
    else:
        self.total_weight_dict = {}

    # Remove old metadata files
    try:
        weights_dir = "vllm_weights_metadata"
        if os.path.exists(weights_dir):
            old_weight_file = os.path.join(weights_dir, f"weights_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")
            old_total_file = os.path.join(weights_dir, f"total_weight_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")

            if os.path.exists(old_weight_file):
                os.remove(old_weight_file)
                # logger.info(f"Removed old weight metadata file: {old_weight_file}")

            if os.path.exists(old_total_file):
                os.remove(old_total_file)
                # logger.info(f"Removed old total weight metadata file: {old_total_file}")

    except Exception as e:
        # logger.warning(f"Failed to clean old metadata files: {e}")
        return

# ===================================================================
# Monkey patch the ModelRunner class methods

def apply_vllm_model_runner_patches():
    print(f"[PATCH] Applying model runner patches in process {os.getpid()}...")
    try:
        from vllm.worker.model_runner import GPUModelRunnerBase

        GPUModelRunnerBase._acquire_weight_lock = _patched_acquire_weight_lock
        GPUModelRunnerBase._release_weight_lock = _patched_release_weight_lock
        GPUModelRunnerBase._register_weight_hooks = _patched_register_weight_hooks
        GPUModelRunnerBase._save_weight_meta = _patched_save_weight_meta
        GPUModelRunnerBase._save_total_weight_meta = _patched_save_total_weight_meta
        GPUModelRunnerBase._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        GPUModelRunnerBase._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(GPUModelRunnerBase, '_original_load_model'):
            GPUModelRunnerBase._original_load_model = GPUModelRunnerBase.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patching GPUModelRunnerBase.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            GPUModelRunnerBase.load_model = patched_load_model
            
    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply GPUModelRunnerBase patches: {e}")
        return

# ====================================================================
# Patch the subprocesses entrypoint functions

def patched_run_worker_process(self) -> None:
    print(f"[VLLM_PATCH_CORE] Patching worker:init_worker in process {os.getpid()} ...")
    apply_vllm_model_runner_patches()

    # from vllm.worker.worker import Worker
    import vllm.executor.multiproc_worker_utils as multiproc_worker_utils_module

    if not hasattr(multiproc_worker_utils_module, '_original_run_worker_process'):
        self._original_run_worker_process = multiproc_worker_utils_module._run_worker_process

    assert hasattr(multiproc_worker_utils_module, '_original_run_worker_process'), "Original multiproc_worker_utils _run_worker_process method not found."

    self._original_run_worker_process()
    

# ===================================================================
# Patch core entrypoint
def apply_entrypoint_patches():
    print(f"[VLLM_PATCH_CORE] Applying entrypoint patches for vLLM server in {os.getpid()} ...")

    try:
        apply_vllm_model_runner_patches()

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply model runner patches: {e}")
        return

    # try:
    #     import vllm.executor.multiproc_worker_utils as multiproc_worker_utils_module

    #     if not hasattr(multiproc_worker_utils_module, '_original_run_worker_process'):
    #         multiproc_worker_utils_module._original_run_worker_process = multiproc_worker_utils_module._run_worker_process

    #     multiproc_worker_utils_module._run_worker_process = patched_run_worker_process

    # except Exception as e:
    #     print(f"[VLLM_PATCH_CORE] Failed to import necessary modules for entrypoint patching: {e}")
    #     return
