import grpc
import time
import uuid
import signal
import sys
import os
import orchestrator_pb2
import orchestrator_pb2_grpc

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Worker:
    def __init__(self, master_address='localhost:50051'):
        self.worker_id = str(uuid.uuid4())[:8]
        self.channel = grpc.insecure_channel(master_address)
        self.stub = orchestrator_pb2_grpc.OrchestratorStub(self.channel)
        self.running = True
        self.rank = None
        self.world_size = None
        self.model = None
        self.optimizer = None
        self.checkpoint_dir = '/tmp/checkpoints'

    def register(self):
        request = orchestrator_pb2.RegisterRequest(worker_id=self.worker_id)
        response = self.stub.Register(request)

        if response.success:
            self.rank = response.assigned_rank
            self.world_size = response.world_size
            print(f"Registered as worker {self.worker_id} with rank {self.rank}")
            print(f"World size: {response.world_size}")
        else:
            print(f"Registration failed: {response.message}")

        return response.success

    def wait_for_training(self):
        print("Waiting for all workers to register...")
        request = orchestrator_pb2.CanStartTrainingRequest(worker_id=self.worker_id)
        response = self.stub.CanStartTraining(request)

        if response.ready:
            self.world_size = response.world_size
            print(f"All workers ready! World size: {self.world_size}")
            return True
        
        return False

    def setup_ddp(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        print(f"Initializing DDP with rank {self.rank}, world_size {self.world_size}")
        dist.init_process_group(
            backend='gloo',
            rank=self.rank,
            world_size=self.world_size
        )
        print("DDP process group initialized")

    def setup_model(self):
        self.model = SimpleModel()
        self.model = DDP(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        print("Model wrapped with DDP")

    def send_heartbeat(self, iteration=0, loss=0.0, is_training=False, checkpointed_iteration=0):
        request = orchestrator_pb2.HeartbeatRequest(
            worker_id=self.worker_id,
            timestamp=int(time.time()),
            current_iteration=iteration,
            current_loss=loss,
            is_training=is_training,
            checkpointed_iteration=checkpointed_iteration
        )
        response = self.stub.SendHeartbeat(request)
        return response

    def save_checkpoint(self, iteration):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'rank_{self.rank}',
            f'checkpoint_iter_{iteration}.pt'
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, iteration):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'rank_{self.rank}',
            f'checkpoint_iter_{iteration}.pt'
        )
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from iteration {iteration}")
        return checkpoint['iteration']

    def train(self, num_iterations=100, start_iteration=0):
        print(f"Starting training from iteration {start_iteration} for {num_iterations} iterations...")
        criterion = nn.MSELoss()

        for iteration in range(start_iteration, num_iterations):
            if not self.running:
                break

            # Generate dummy data (same seed across workers for sync)
            torch.manual_seed(iteration)
            inputs = torch.randn(32, 10)
            targets = torch.randn(32, 1)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # Send heartbeat with training progress
            response = self.send_heartbeat(
                iteration=iteration + 1,
                loss=loss_val,
                is_training=True
            )

            # Check if master requested checkpoint
            if response.should_checkpoint:
                self.save_checkpoint(iteration + 1)
                # Immediately signal completion
                self.send_heartbeat(
                    iteration=iteration + 1,
                    loss=loss_val,
                    is_training=True,
                    checkpointed_iteration=iteration + 1
                )

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss_val:.4f}")

            time.sleep(0.5)  # Slow down for demo

        print("Training completed!")

    def check_for_recovery(self):
        request = orchestrator_pb2.RecoveryRequest(worker_id=self.worker_id)
        response = self.stub.GetRecoveryInfo(request)
        return response

    def run(self):
        start_iteration = 0
        
        if not self.register():
            return

        if not self.wait_for_training():
            return

        try:
            self.setup_ddp()
            self.setup_model()

            recovery_info = self.check_for_recovery()
            if recovery_info.in_recovery_mode:
                print("Master is in recovery mode. Attempting to load checkpoint.")
                start_iteration = self.load_checkpoint(recovery_info.checkpoint_iteration)

            self.train(
                num_iterations=int(os.environ.get('NUM_ITERATIONS', 100)),
                start_iteration=start_iteration
            )
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
            print("DDP process group destroyed")

    def stop(self):
        print(f"\nWorker {self.worker_id} shutting down...")
        self.running = False
        self.cleanup()
        self.channel.close()


def main():
    worker = Worker()

    def signal_handler(sig, frame):
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    worker.run()


if __name__ == '__main__':
    main()
