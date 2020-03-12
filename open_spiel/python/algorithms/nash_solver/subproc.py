import subprocess
import time
import signal
import os
import logging

"""
This script provides subprocess API for gambit solver.
"""

def call_and_wait_with_timeout(command_str, timeout):
    logging.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True, preexec_fn=os.setsid)
    timeout_seconds = timeout
    try:
        my_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logging.info("Process ran more seconds than: " + str(timeout_seconds))
        os.killpg(os.getpgid(my_process.pid), signal.SIGTERM)
        logging.info("Subprocess has been killed.")
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()

def call_and_wait(command_str):
    logging.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True)
    my_process.wait()
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()

def call_and_wait_with_timeout_and_check(command_str):
    logging.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True)
    timeout_seconds = 3600
    try:
        my_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logging.info("Process ran more seconds than: " + str(timeout_seconds))
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()