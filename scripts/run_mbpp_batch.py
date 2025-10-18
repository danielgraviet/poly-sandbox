import asyncio
from evaluators.benchmark import run_mbpp_batch

if __name__ == "__main__":
    asyncio.run(run_mbpp_batch(n=120, concurrency=5))
