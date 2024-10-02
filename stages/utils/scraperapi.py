import queue
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional
import threading

@dataclass
class ScrapeTask:
    url: str
    autoparse: bool = False
    binary: bool = False
    ultra_premium: bool = False

class WebCrawler:
    def __init__(self, num_workers: int, scraperapi_key: str, max_queue_size: int):
        self.num_workers = num_workers
        self.scraperapi_key = scraperapi_key
        self.task_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.semaphore = threading.Semaphore(max_queue_size)

    def start(self):
        for _ in range(self.num_workers):
            self.executor.submit(self._worker)

    def _worker(self):
        while True:
            future, task = self.task_queue.get()
            try:
                result = self._scrape(task)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.task_queue.task_done()
                self.semaphore.release()

    def _scrape(self, task: ScrapeTask) -> requests.Response:
        params = {
            'api_key': self.scraperapi_key,
            'url': task.url,
            'autoparse': task.autoparse,
            'binary_target': task.binary,
            'ultra_premium': task.ultra_premium
        }
        return requests.get('http://api.scraperapi.com', params=params)

    def crawl(self, url: str, autoparse: bool = False, binary: bool = False, ultra_premium: bool = False) -> Future:
        task = ScrapeTask(url, autoparse, binary, ultra_premium)
        future = Future()
        self.semaphore.acquire()  # This will block if the semaphore count is at 0
        self.task_queue.put((future, task))
        return future

    def shutdown(self):
        self.executor.shutdown(wait=True)