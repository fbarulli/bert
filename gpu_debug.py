import gc
import torch
import wandb
import psutil
import os
import time
import subprocess
from typing import Dict, Any, Optional, List
import numpy as np

class GPUMemoryDebugger:
    def __init__(self):
        self.snapshots = []
        
    def take_snapshot(self, label: str):
        snapshot = {
            'label': label,
            'time': time.time(),
            'tensors': [],
            'memory': self._get_memory_stats(),
            'wandb': self._get_wandb_state(),
            'cuda_context': self._get_cuda_context(),
            'gpu_processes': self._get_gpu_processes()
        }
        
        # Get all CUDA tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_info = {
                        'shape': tuple(obj.shape),
                        'dtype': str(obj.dtype),
                        'size_mb': obj.element_size() * obj.nelement() / (1024*1024),
                        'device': str(obj.device),
                        'requires_grad': obj.requires_grad
                    }
                    snapshot['tensors'].append(tensor_info)
            except:
                pass
                
        self.snapshots.append(snapshot)
        
    def _get_nvidia_smi_stats(self) -> Dict[str, float]:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', '--format=csv,nounits,noheader']
            )
            used, total, free = map(float, result.decode('utf-8').strip().split(','))
            return {
                'nvidia_used_mb': used,
                'nvidia_total_mb': total,
                'nvidia_free_mb': free
            }
        except:
            return {}
    
    def _get_gpu_processes(self) -> List[Dict[str, Any]]:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', 'pmon', '-c', '1'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8')
            
            processes = []
            for line in result.strip().split('\n')[2:]:  # Skip header lines
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        pid = parts[1]
                        if pid != '-':
                            try:
                                process = psutil.Process(int(pid))
                                processes.append({
                                    'pid': pid,
                                    'name': process.name(),
                                    'cmdline': ' '.join(process.cmdline()),
                                    'gpu_memory': parts[3] if parts[3] != '-' else '0'
                                })
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            return processes
        except:
            return []
    
    def _get_cuda_context(self) -> Dict[str, Any]:
        try:
            import torch.cuda
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
                'is_initialized': torch.cuda.is_initialized(),
                'has_memory_allocated': bool(torch.cuda.memory_allocated()),
                'has_memory_cached': bool(torch.cuda.memory_reserved())
            }
        except:
            return {}
        
    def _get_memory_stats(self) -> Dict[str, float]:
        stats = {
            'cuda_allocated': torch.cuda.memory_allocated() / (1024**2),
            'cuda_reserved': torch.cuda.memory_reserved() / (1024**2),
            'cuda_max_allocated': torch.cuda.max_memory_allocated() / (1024**2),
            'system_used': psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        }
        
        # Add nvidia-smi stats
        stats.update(self._get_nvidia_smi_stats())
        
        if hasattr(torch.cuda, 'memory_stats'):
            cuda_stats = torch.cuda.memory_stats()
            stats.update({
                'active_allocs': cuda_stats.get('num_alloc_retries', 0),
                'active_bytes': cuda_stats.get('allocated_bytes.all.current', 0) / (1024**2)
            })
        
        return stats
        
    def _get_wandb_state(self) -> Dict[str, Any]:
        try:
            if not wandb.run:
                return {'active': False}
                
            return {
                'active': True,
                'name': wandb.run.name,
                'mode': wandb.run.mode,
                'finished': wandb.run.finished
            }
        except:
            return {'active': False}
        
    def print_snapshot(self, label: Optional[str] = None):
        """Print specific snapshot or latest if no label provided"""
        snapshot = next(
            (s for s in reversed(self.snapshots) if label is None or s['label'] == label),
            None
        )
        
        if not snapshot:
            print(f"No snapshot found{f' with label {label}' if label else ''}")
            return
            
        print(f"\n=== Memory Snapshot: {snapshot['label']} ===")
        
        print("\nCUDA Memory (PyTorch):")
        for k, v in snapshot['memory'].items():
            if 'cuda' in k:
                print(f"- {k}: {v:.1f}MB")
                
        print("\nNvidia-SMI Memory:")
        for k, v in snapshot['memory'].items():
            if 'nvidia' in k:
                print(f"- {k}: {v:.1f}MB")
        
        print("\nCUDA Context:")
        for k, v in snapshot['cuda_context'].items():
            print(f"- {k}: {v}")
                
        print("\nTensors on GPU:", len(snapshot['tensors']))
        total_mb = sum(t['size_mb'] for t in snapshot['tensors'])
        print(f"Total tensor memory: {total_mb:.1f}MB")
        
        if snapshot['tensors']:
            print("\nLargest tensors:")
            for t in sorted(snapshot['tensors'], key=lambda x: x['size_mb'], reverse=True)[:5]:
                print(f"- Shape: {t['shape']}, Type: {t['dtype']}, Size: {t['size_mb']:.1f}MB")
        
        if snapshot['gpu_processes']:
            print("\nProcesses using GPU:")
            for proc in snapshot['gpu_processes']:
                print(f"- PID {proc['pid']}: {proc['name']} ({proc['gpu_memory']}MB)")
                print(f"  {proc['cmdline']}")
                
        if snapshot['wandb']['active']:
            print("\nWandb Status:")
            print(f"- Run: {snapshot['wandb']['name']}")
            print(f"- Mode: {snapshot['wandb']['mode']}")
            print(f"- Finished: {snapshot['wandb']['finished']}")
            
    def compare_snapshots(self, label1: str, label2: str):
        """Compare two snapshots to see what changed"""
        snap1 = next((s for s in self.snapshots if s['label'] == label1), None)
        snap2 = next((s for s in self.snapshots if s['label'] == label2), None)
        
        if not snap1 or not snap2:
            print("Snapshots not found")
            return
            
        print(f"\n=== Comparing {label1} vs {label2} ===")
        
        # Compare memory stats
        print("\nMemory Changes:")
        for k in snap1['memory'].keys():
            if 'cuda' in k or 'nvidia' in k:
                diff = snap2['memory'].get(k, 0) - snap1['memory'].get(k, 0)
                print(f"- {k}: {diff:+.1f}MB")
                
        # Compare tensor counts
        tensors1 = len(snap1['tensors'])
        tensors2 = len(snap2['tensors'])
        print(f"\nTensor count: {tensors1} -> {tensors2} ({tensors2-tensors1:+d})")
        
        # Compare total memory
        total1 = sum(t['size_mb'] for t in snap1['tensors'])
        total2 = sum(t['size_mb'] for t in snap2['tensors'])
        print(f"Total tensor memory: {total1:.1f}MB -> {total2:.1f}MB ({total2-total1:+.1f}MB)")
        
        # Compare processes
        procs1 = {p['pid']: p for p in snap1['gpu_processes']}
        procs2 = {p['pid']: p for p in snap2['gpu_processes']}
        
        new_pids = set(procs2.keys()) - set(procs1.keys())
        gone_pids = set(procs1.keys()) - set(procs2.keys())
        
        if new_pids or gone_pids:
            print("\nProcess Changes:")
            if new_pids:
                print("New processes:")
                for pid in new_pids:
                    p = procs2[pid]
                    print(f"+ PID {p['pid']}: {p['name']} ({p['gpu_memory']}MB)")
            
            if gone_pids:
                print("Terminated processes:")
                for pid in gone_pids:
                    p = procs1[pid]
                    print(f"- PID {p['pid']}: {p['name']} ({p['gpu_memory']}MB)")

def force_cuda_reset():
    """Attempt to force CUDA context reset"""
    try:
        # Delete all PyTorch CUDA tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except:
                pass
        
        # Clear memory caches
        gc.collect()
        torch.cuda.empty_cache()
        
        # Try to reset CUDA
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        # Force device synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Attempt to reset device
        try:
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
        except:
            pass
            
    except Exception as e:
        print(f"Error during CUDA reset: {e}")

if __name__ == "__main__":
    # Create debugger instance
    debugger = GPUMemoryDebugger()
    
    print("=== Initial State ===")
    debugger.take_snapshot("initial")
    debugger.print_snapshot("initial")
    
    print("\n=== Attempting Normal Cleanup ===")
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n=== After Normal Cleanup ===")
    debugger.take_snapshot("after_cleanup")
    debugger.print_snapshot("after_cleanup")
    
    print("\n=== Attempting Force Reset ===")
    force_cuda_reset()
    
    print("\n=== After Force Reset ===")
    debugger.take_snapshot("after_reset")
    debugger.print_snapshot("after_reset")
    
    print("\n=== Changes ===")
    debugger.compare_snapshots("initial", "after_reset")
