# Demo visualization


## Installation
The following packages are needed to run the demo server:
```
pip install openai dash dash-bootstrap-components dash-extensions quart atomicwrites loguru
```

If you it a dependency issue in pip, you could also try installing the heavier libs from conda
```
mamba install -c conda-forge dash dash-bootstrap-components dash-extensions quart
pip install openai loguru atomicwrites
```

## Running
The demo expects to run from src/projects/scannet_offline_eval/demo, and listens for updates to `./publishers/published_trajectory`. The configuration settings are defined in `components/app.py`. The server and all components load `components.app`, which acts as shared state.

To run the demo server with a canned trajectory, create or copy the trajectory folder to the appropriate locaton, and run `python server.py`. This will start the server (default port 8901)

A full script would be something like:
```
# Remote terminal window 1
python server.py

# Local terminal window 1 (if running remotely)
ssh -N -L 8901:localhost:8901 remote_machine_url
```

Then you can point your browser to localhost:8901.

#### Using websockets instead of server-side callbacks
In `app.py` there are some commented out lines when creating the gripper feed UI element. The commented lines are an alternative implementation that uses websockets instead of server-side callbacks. Websockets are lower-latency and make fewer (expensive) requrests to the server. Websockets (and also server-side events) allow us to push notifications to the client instead of asking the client to periodically poll the server for updates.

In our case, it doesn't make much of a performance difference -- but the websocket implementation is there and it works. We can use that as a reference elsewhere. For example, if we wanted to change the communication implementation for the realtime-3d UI element. 

To run the websocket implementation, you need to also start a second (Quart) server that provides websocket connections. A full script would be something like:
```
# Remote terminal window 1
python -m components.publisher_server # Creates a DirectoryWatcher server that publishes with a websocket at port 5000

# Remote terminal window 2
python server.py # listens on port 8901

# Local terminal window 1 (if running remotely)
ssh -N -L 8901:localhost:8901 remote_machine_url
```

