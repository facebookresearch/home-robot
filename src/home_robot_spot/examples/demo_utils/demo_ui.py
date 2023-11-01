import subprocess

import psutil


def start_demo_ui_server():
    """Start the demo UI server as a separate process."""

    subprocess.Popen(["python", "server.py"], cwd="projects/scannet_offline_eval/demo/")


def stop_demo_ui_server():
    """Stop the demo UI server."""

    # Find the process by its command line
    for proc in psutil.process_iter(["cmdline"]):
        if proc.info["cmdline"] == ["python", "server.py"]:
            proc.kill()
