import grpc
import time
import uuid
import signal
import sys
import orchestrator_pb2
import orchestrator_pb2_grpc


class Worker:
    def __init__(self, master_address='localhost:50051'):
        self.worker_id = str(uuid.uuid4())[:8]
        self.channel = grpc.insecure_channel(master_address)
        self.stub = orchestrator_pb2_grpc.OrchestratorStub(self.channel)
        self.running = True
        self.rank = None

    def register(self):
        request = orchestrator_pb2.RegisterRequest(worker_id=self.worker_id)
        response = self.stub.Register(request)

        if response.success:
            self.rank = response.assigned_rank
            print(f"Registered as worker {self.worker_id} with rank {self.rank}")
            print(f"World size: {response.world_size}")
        else:
            print(f"Registration failed: {response.message}")

        return response.success

    def send_heartbeat(self):
        request = orchestrator_pb2.HeartbeatRequest(
            worker_id=self.worker_id,
            timestamp=int(time.time())
        )
        response = self.stub.SendHeartbeat(request)
        print(f"Heartbeat sent, acknowledged: {response.acknowledged}")

    def run(self):
        if not self.register():
            return

        print("Starting heartbeat loop (Ctrl+C to stop)...")
        while self.running:
            try:
                self.send_heartbeat()
                time.sleep(5)
            except grpc.RpcError as e:
                print(f"RPC error: {e}")
                break

    def stop(self):
        print(f"\nWorker {self.worker_id} shutting down...")
        self.running = False
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
