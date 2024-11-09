import asyncio
import time

async def task1():
    start_time = time.time()
    print('Task 1 started')
    await asyncio.sleep(1)
    time.sleep(0)
    end_time = time.time()
    print(f'Task 1 ended, elapsed time: {end_time - start_time:.2f} seconds')

async def task2():
    start_time = time.time()
    print('Task 2 started')
    await asyncio.sleep(2)
    time.sleep(0)
    end_time = time.time()
    print(f'Task 2 ended, elapsed time: {end_time - start_time:.2f} seconds')

async def main():
    task1_coro = asyncio.create_task(task1())
    task2_coro = asyncio.create_task(task2())
    await asyncio.sleep(8)
    await task1_coro
    print('Task 1 completed')
    await task2_coro
    print('All tasks completed')

if __name__ == "__main__":
    asyncio.run(main())