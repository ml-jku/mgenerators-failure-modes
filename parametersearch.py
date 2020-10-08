"""
Distributed Parameter Search
============================
This library distributes parameter searches over many clients.
Its designed for simplicity and hackability first.
It's author is Thomas Unterthiner.

Simple Usage
------------

This is a single-file implementation, so just copy ```parametersearch.py``` to your source directory.
From there, do `from parametersearch import ParameterSearch` to use it.
`ParameterSearch` can be used to define all the different hyperparameter settings you want to try out.
As example, this piece of code defines two settings of different learning rates:

    ps = ParameterSearch(output_file="results.csv")  # results will be stored in results.csv
    ps.add_parameter_setting({"learning_rate": 1e-2})
    ps.add_parameter_setting({"learning_rate": 1e-3})

or you can use ```define_search_grid``` to set up a grid search:

    param_grid = [{
        'n_estimators': [20, 50],
        'max_features': [14, 28]
        }]
    ps = define_search_grid(param_grid, output_file="results.csv")


Then, you can iterate over the created ParameterSearch instance to process the different settings, and
use the ```submit_result``` method to report the results back to the ParameterSearch object:

    for (job_id, hyperparams) in ps:
        print("Working on job %d: %s" % (job_id, hyperparams), flush=True)
        model = sklearn.ensemble.RandomForestClassifier(**hyperparams)
        model.fit(x_tr, y_tr)
        p_va = model.predict(x_va)
        accuracy_va = metrics.accuracy_score(y_va, p_va)
        ps.submit_result(job_id, accuracy_va)


Distributed Usage
-----------------
You can distribute your hyperparameter search over several machines. To do this, set up your ParameterSearch
as usual in your server process, then call ```ParameterSearch.start_server(...)``` to make your
hyperparameter search available to the outside world.

Next start up any client processes: these create ParameterSearch instances that connect to the server process:

    ps = ParameterSearch(host="my.server.com", port=5732)

And then use the ParameterSearch as usual. It will connect to the server and receive parameter settings defined
there. See ```example.py``` for a simple example.


License
-------
Distributed Parameter Search is copyrighted (c) 2019 by Thomas Unterthiner and licensed under the
`General Public License (GPL) Version 2 or higher <http://www.gnu.org/licenses/gpl-2.0.html>`_.
See ``LICENSE.md`` for the full details.
"""


import csv
import json
import logging
import socket
import struct
import socketserver
import threading
from sklearn.model_selection import ParameterGrid
import signal

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def define_search_grid(param_grid, output_file=None):
    """Creates a grid search.

    param_grid: specifies the points in the grid. This uses the same format as GridSearchCV in sklearn, see
                https://scikit-learn.org/stable/modules/grid_search.html#grid-search
    output_file: A CSV file that will be used to store the results of a hyperparameter search (optional)
    Returns: a ParameterSearch object
    """
    m = ParameterSearch(output_file=output_file)
    for param_setting in ParameterGrid(param_grid):
        m.add_parameter_setting(param_setting)
    return m


class ParameterSearch(object):
    """A ParameterSearch stores the settings that will be tried out, as well as their eventual results.

    A ParameterSearch can work over a network, with one instance acting as server that hands out parameter
    settings and accepts results from remote client instances.

    Note: settings are not stored in order, i.e., the order in which you add them to the ParameterSearch does not
    necessarily coincide with the order in which they are handed out.
    """

    def __init__(self, host=None, port=None, output_file=None):
        """Creates a new ParameterSearch instance.

        If this is a client instance that gets its parameters from a remote instance, specify their host/port here.

        host: host name or IP address of server ParameterSearch instance (optional, also requires a port)
        port: port of server ParameterSearch instance port (optional, also requires a host)
        output_file: A CSV file that will be used to store the results of a hyperparameter search
        """
        self.waiting_jobs = []
        self.running_jobs = []
        self.working_jobs = []
        self.log = logging.getLogger('dipasearch')
        self.log.setLevel(logging.INFO)
        self.is_serving = False
        if host is not None and port is None:
            raise RuntimeError("passed address but no port")
        elif port is not None and host is None:
            raise RuntimeError("passed port but no address")
        if host is not None and port is not None and output_file is not None:
            raise RuntimeError("client instances cannot store output files")

        self.is_client = host is not None and port is not None
        if not self.is_client:
            self.database = Database(output_file)
            self.database_lock = threading.Lock()
        else:
            self.host = socket.gethostbyname(host)
            self.port = port


    def add_parameter_setting(self, setting):
        """Adds a setting to the search.

        setting: a dictionary that maps setting-names to the values they take
        """
        job = self.database.add_job(setting)
        self.waiting_jobs.append(job.id)

    def start_server(self, host, port, as_thread=False):
        """Starts accepting remote requests for jobs and waits for replies.

        host: the IP address or hostname from which to serve from
        port: the port from which to serve from
        as_thread: if true, start the server in a separate thread
        """

        assert not self.is_client, "Clients cannot act as Servers"
        self.log.info('Starting up server on %s:%d' % (host, port))
        self.is_serving = True

        def _server_loop(host_, port_, param_search_server):
            """The event loop for the server thread"""
            h = socket.gethostbyname(host_)
            socketserver.ThreadingTCPServer.allow_reuse_address = True
            with socketserver.ThreadingTCPServer((h, port_), _ServerRequestHandler(param_search_server)) as server:
                server.timeout = 1  # seconds until we'll check if we need to stop serving
                while param_search_server.is_serving:
                    server.handle_request()
        if as_thread:
            t = threading.Thread(target=_server_loop, args=(host, port, self, ))
            t.start()
        else:
            _server_loop(host, port, self)

    def __iter__(self):
        return self

    def __next__(self):
        job_id, params = self.get_next_setting()
        if job_id is None:
            raise StopIteration()
        else:
            return job_id, params

    def get_results(self):
        assert not self.is_client, "Clients don't have access to the result list"
        return list(self.database.get_all_jobs())

    def get_next_setting(self):
        """Gets the next untried hyperparameter setting.

        Optionally, the setting can be requested from a remote ParameterSearch instance running on another
        host/port.

        Returns:
            a pair of job_id and the setting to try out, or (None, None) if there are no more settings
        """

        if self.is_client:
            job_id, data = self._request_remote_parameter_set()
            if job_id is not None:
                self.working_jobs.append(job_id)
            return job_id, data

        if not self.waiting_jobs:
            return None, None

        with self.database_lock:
            job_id = self.waiting_jobs.pop(0)
            self.running_jobs.append(job_id)
            job = self.database.get_job(job_id)
        return job.id, job.data

    def submit_result(self, job_id, result):
        """
        Submits the results of a job.
        """

        if self.is_client:
            if job_id not in self.working_jobs:
                raise RuntimeError(f'This client is not working on job {job_id}')
            
            self._submit_remote_job(job_id, result)
            self.working_jobs.remove(job_id)
            return

        with self.database_lock:
            if job_id not in self.running_jobs:
                self.log.info(f"submission rejected, job {job_id}: Job not running.")
                raise RuntimeError("Job not running.")
            
            self.running_jobs.remove(job_id)

            if result == 0:
                self.database.complete_job(job_id, result)
            else:
                self.log.info(f"job failed, job {job_id}: reattaching to waiting jobs")
                self.waiting_jobs.append(job_id)
            

        print('Running: ', self.running_jobs)
        print('Waiting: ', self.waiting_jobs)
        if not self.running_jobs and not self.waiting_jobs:
            self.log.info("All jobs finished, sending server shutdown signal")
            self.is_serving = False

    def _submit_remote_job(self, job_id, result):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            data = {'id': job_id, 'result': result}
            sock.connect((self.host, self.port))
            sock.send(b'S')
            data = json.dumps(data).encode("utf8")
            self.log.info("submitting data: %s" % data)
            sock.send(struct.pack("<I", len(data)))
            sock.sendall(data)
            is_ok = struct.unpack("b", sock.recv(1))[0]
            if not is_ok:
                with sock.makefile() as f:
                    error_msg = f.read()
                raise RuntimeError("Result submission failed: %s" % error_msg)
        return

    def _request_remote_parameter_set(self, retries=0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.send(b'R')
                with sock.makefile() as f:
                    data = f.read()
            if data is None or not data:
                return None, None
            d = json.loads(data)
            job_id = d['id']
            data = d['data']
            return job_id, data
        except ConnectionRefusedError as err:
            self.log.warning("connection refused: %s (retries: %d)" % (err, retries))
            if retries > 0:
                time.sleep(1)
                return self._request_remote_parameter_set(retries-1)
            else:
                return None, None


class _ServerRequestHandler(socketserver.StreamRequestHandler):
    """Handles the requests of a ParameterSearch server instance"""
    def __init__(self, parameter_search):
        self.parameter_search = parameter_search

    # see https://stackoverflow.com/questions/15889241/send-a-variable-to-a-tcphandler-in-python
    def __call__(self, request, client_address, server):
        h = _ServerRequestHandler(self.parameter_search)
        socketserver.StreamRequestHandler.__init__(h, request, client_address, server)

    def handle(self):
        msgtype = self.rfile.read(1)
        if msgtype == b'R':  # request job
            job_id, job_data = self.parameter_search.get_next_setting()
            if job_id is None:
                self.parameter_search.log.warning("no jobs left for current request")
                return
            self.parameter_search.log.info("new request, sending job %d" % job_id)
            data = {'id': job_id, 'data': job_data}
            data = json.dumps(data).encode("utf8")
            self.wfile.write(data)
        elif msgtype == b"S":  # finished a job
            self.parameter_search.log.debug("preparing to receive submission")
            buflen = struct.unpack("<I", self.rfile.read(4))[0]
            data = self.rfile.read(buflen).decode("utf8")
            data = json.loads(data)
            self.parameter_search.log.info("new submission, job %d, result: %s" % (data['id'], data['result']))
            try:
                self.parameter_search.submit_result(data['id'], data['result'])
            except RuntimeError as err:
                self.wfile.write(struct.pack("b", 0))
                self.wfile.write(str(err).encode("utf8"))
            else:
                self.wfile.write(struct.pack("b", 1))
            self.wfile.flush()
        else:
            self.parameter_search.error("Unknown message type: %s" % msgtype)
            raise RuntimeError("Unknown message type: %s" % msgtype)



class ParameterClient(object):
    def __init__(self, host=None, port=None):
        """Creates a new ParameterSearch instance.

        If this is a client instance that gets its parameters from a remote instance, specify their host/port here.

        host: host name or IP address of server ParameterSearch instance (optional, also requires a port)
        port: port of server ParameterSearch instance port (optional, also requires a host)
        output_file: A CSV file that will be used to store the results of a hyperparameter search
        """
        self.working_jobs = []
        self.log = logging.getLogger('dipasearch')
        self.log.setLevel(logging.INFO)
        
        # we are client
        self.is_serving = False
        
        if host is not None and port is None:
            raise RuntimeError("passed address but no port")
        elif port is not None and host is None:
            raise RuntimeError("passed port but no address")

        self.is_client = True
        self.host = socket.gethostbyname(host)
        self.port = port

    def __iter__(self):
        return self

    def __next__(self):
        job_id, params = self.get_next_setting()
        if job_id is None:
            raise StopIteration()
        else:
            return job_id, params

    def get_next_setting(self):
        """Gets the next untried hyperparameter setting.

        Optionally, the setting can be requested from a remote ParameterSearch instance running on another
        host/port.

        Returns:
            a pair of job_id and the setting to try out, or (None, None) if there are no more settings
        """

        if self.is_client:
            job_id, data = self._request_remote_parameter_set()
            if job_id is not None:
                self.working_jobs.append(job_id)
            return job_id, data
        
class _Job(object):
    """A Job is the internal representation of a hyperparameter setting."""
    def __init__(self, id, data, result):
        self.id = id
        self.data = data
        self.result = result


class Database(object):
    """This class stores and manages all past and current _Job instances."""

    def __init__(self, output_file=None):
        self.data = {}
        self._max_idx = 1
        self.output_file = output_file

    def add_job(self, job_data):
        job_id = self._max_idx
        self.data[job_id] = _Job(job_id, job_data, None)
        self._max_idx += 1
        return self.data[job_id]

    @property
    def n_jobs(self):
        return len(self.data)

    def get_job(self, job_id):
        return self.data[job_id]

    def complete_job(self, job_id, result):
        if job_id not in self.data:
            raise RuntimeError("Job does not exist")
        job = self.data[job_id]
        if job.result is not None:
            raise RuntimeError("Job already completed")
        job.result = result
        self._save_results()

    def _save_results(self):
        if self.output_file is None:
            return
        with open(self.output_file, "w", newline='') as f:
            fieldnames = ["id", "parameters", "result"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for j in self.data.values():
                params = {"id": j.id, "parameters": json.dumps(j.data), "result": j.result}
                writer.writerow(params)

    def get_all_jobs(self):
        for j in self.data.values():
            yield (j.id, j.data, j.result)
