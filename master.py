from orchestrator_pb2_grpc import OrchestratorServicer
import orchestrator_pb2
import grpc
import orchestrator_pb2_grpc
from concurrent import futures

class OrchestratorServicerMaster(OrchestratorServicer):
    def __init__(self):
        self.workers = {}
        self.next_rank = 0
    
    def Register(self, request, context):
        worker_id = request.worker_id

        if worker_id in self.workers:
            return orchestrator_pb2.RegisterResponse(
                assigned_rank=self.workers[worker_id]['rank'],
                world_size=len(self.workers),
                success=False,
                message=f"Worker {worker_id} already registered"
            )
        
        assigned_rank = self.next_rank
        self.next_rank += 1

        self.workers[worker_id] = {
            'rank': assigned_rank,
        }

        print(f"Worker {worker_id} registered with rank {assigned_rank}")

        return orchestrator_pb2.RegisterResponse(
            assigned_rank=assigned_rank,
            world_size=len(self.workers),
            success=True,
            message="Registration successful"
        ) 
    
    def SendHeartbeat(self, request, context):
        worker_id = request.worker_id
        timestamp = request.timestamp

        print(f"Worker {worker_id} sent heartbeat at timestamp {timestamp}")

        return orchestrator_pb2.HeartbeatResponse(acknowledged=True)
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestrator_pb2_grpc.add_OrchestratorServicer_to_server(
        OrchestratorServicerMaster(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Orchestrator server started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
