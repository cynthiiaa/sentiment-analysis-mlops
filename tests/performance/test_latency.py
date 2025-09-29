import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.inference import app
from src.models.sentiment_model import SentimentModel


class TestPerformance:
    @pytest.fixture
    def model(self):
        return SentimentModel()

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_single_prediction_latency(self, model):
        """Test single prediction latency"""
        text = "This is a test sentence for performance measurement."

        latencies = []
        for _ in range(100):
            start = time.time()
            _ = model.predict([text])
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print("\nSingle Prediction Latency")
        print(f"    Average: {avg_latency*1000:.2f}ms")
        print(f"    P95: {p95_latency*1000:.2f}ms")
        print(f"    P99: {p99_latency*1000:.2f}ms")

        # assert performance requirements
        assert avg_latency < 0.1, "Average latency exceeds 100ms"
        assert p95_latency < 0.2, "P95 latency exceeds 200ms"

    def test_batch_processing_throughput(self, model):
        """Test batch processing throughput"""
        texts = [f"Test sentence number {i}" for i in range(100)]

        start = time.time()
        _ = model.predict(texts)
        total_time = time.time() - start

        throughput = len(texts) / total_time

        print("\nBatch Processing:")
        print(f"    Samples: {len(texts)}")
        print(f"    Total Time: {total_time:.2f}s")
        print(f"    Throughput: {throughput:.2f} samples/second")

        assert throughput > 50, "Throughput below 50 samples/second"

    def test_concurrent_requests(self, client):
        """Test API under concurrent load"""

        def make_requests():
            response = client.post(
                "/predict",
                json={"texts": ["Test concurrent request"], "return_probabilities": False},
            )
            return response.status_code == 200

        num_requests = 50
        with ThreadPoolExecutor(max_workers=10) as executor:
            start = time.time()
            futures = [executor.submit(make_requests) for _ in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
            total_time = time.time() - start

        success_rate = sum(results) / len(results)
        requests_per_second = num_requests / total_time

        print("\nConcurrent Requests:")
        print(f"    Total Requests: {num_requests}")
        print(f"    Success Rate: {success_rate*100:.1f}%")
        print(f"    Requests/Second: {requests_per_second:.2f}")

        assert success_rate >= 0.99, "Success rate below 99%"
        assert requests_per_second > 20, "RPS below 20"

    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async API performance"""

        async def fetch(session, url, data):
            async with session.post(url, json=data) as response:
                return await response.json()

        url = "http://localhost:8000/predict"
        data = {"texts": ["Async test"], "return_probabilities": False}

        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [fetch(session, url, data) for _ in range(100)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start

        avg_time_per_request = total_time / len(results) * 1000

        print("Async Performance:")
        print(f"    Requests: {len(results)}")
        print(f"    Total Time: {total_time:.2f}s")
        print(f"    Avg Time/Requests: {avg_time_per_request:.2f}ms")

        assert avg_time_per_request < 50, "Average async request time exceeds 50ms"

    def test_memory_usage(self, model):
        """Test memory consumption"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # process large batch
        large_texts = ["Test sentence" * 50 for _ in range(100)]
        _ = model.predict(large_texts)

        # peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory

        print("\nMemory Usage")
        print(f"    Baseline: {baseline_memory:.2f}MB")
        print(f"    Peak: {peak_memory:.2f}MB")
        print(f"    Increase: {memory_increase:.2f}MB")

        assert memory_increase < 500, "Memory increase exceeds 500MB"

    @pytest.mark.parametrize("text_length", [10, 50, 100, 200, 500])
    def test_latency_vs_text_length(self, model, text_length):
        """Test how text length affects latency"""
        text = " ".join(["word"] * text_length)

        latencies = []
        for _ in range(20):
            start = time.time()
            _ = model.predict([text])
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies) * 1000  # convert to ms

        print(f"\nText Length {text_length} words: {avg_latency:.2f}ms")

        # latency should scale reasonably with text length
        assert avg_latency < text_length * 2, f"Latency too high for {text_length} words"
