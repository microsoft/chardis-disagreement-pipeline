from tqdm import tqdm
import concurrent.futures as cf
import multiprocessing


def get_default_num_threads():
    """
    Returns a safe default number of threads based on the machine's CPU count.

    Strategy:
    - Use (CPU cores - 1) to avoid maxing out the system.
    - Always return at least 1.
    """
    try:
        cores = multiprocessing.cpu_count()
        return max(1, cores - 1)
    except Exception:
        # Fallback: always safe
        return 1


def execute_func_in_parallel_with_status(items, item_func, num_threads, description) -> list:
    """
    Executes a function in parallel on a list of items with a progress bar.

    Args:
        items (list): The list of items to process.
        item_func (function): The function to execute for each item. It should take a single item as an argument.
        num_threads (int): The number of threads to use for parallel execution.
        description (str): The description to display on the progress bar.

    Returns:
        list: A list of results returned by the item_func for each item.
    """
    task_results = list()

    with tqdm(total=len(items), desc=description) as progress_bar:
        with cf.ThreadPoolExecutor(max_workers=max(num_threads, 1)) as executor:
            result_tasks = {executor.submit(item_func, item): index for index, item in enumerate(items)}
            ordered = {}

            for task in cf.as_completed(result_tasks):
                ordered[result_tasks[task]] = task.result()
                progress_bar.update(1)
            task_results = [ordered[index] for index in sorted(ordered)]

    return task_results


def execute_func_in_parallel(items, item_func, num_threads):
    """
    Executes a function in parallel on a list of items.

    Args:
        items (list): The list of items to process.
        item_func (function): The function to execute for each item. It should take a single item as an argument.
        num_threads (int): The number of threads to use for parallel execution.

    Returns:
        Iterator: An iterator of results returned by the item_func for each item.
    """
    with cf.ThreadPoolExecutor(max_workers=max(num_threads, 1)) as executor:
        result_tasks = [executor.submit(item_func, item) for item in items]

        return cf.as_completed(result_tasks)


def execute_action_in_parallel_with_status(items, item_action, num_threads, description):
    """
    Executes an action in parallel on a list of items with a progress bar.

    Args:
        items (list): The list of items to process.
        item_action (function): The action to execute for each item. It should take a single item as an argument.
        num_threads (int): The number of threads to use for parallel execution.
        description (str): The description to display on the progress bar.
    """
    with tqdm(total=len(items), desc=description) as progress_bar:
        with cf.ThreadPoolExecutor(max_workers=max(num_threads, 1)) as executor:
            result_tasks = [executor.submit(
                item_action, item) for item in items]

            for task in cf.as_completed(result_tasks):
                progress_bar.update(1)


def execute_action_in_parallel(items, item_action, num_threads):
    """
    Executes an action in parallel on a list of items.

    Args:
        items (list): The list of items to process.
        item_action (function): The action to execute for each item. It should take a single item as an argument.
        num_threads (int): The number of threads to use for parallel execution.
    """
    with cf.ThreadPoolExecutor(max_workers=max(num_threads, 1)) as executor:
        result_tasks = [executor.submit(item_action, item) for item in items]

        for task in cf.as_completed(result_tasks):
            pass
