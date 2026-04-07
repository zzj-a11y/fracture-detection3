"""
Gunicorn configuration for fracture detection system
Optimized for Render.com with 4GB memory
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:" + os.environ.get("PORT", "8000")
backlog = 2048

# Worker processes
workers = 1  # Only 1 worker due to memory constraints
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120  # 2 minutes timeout
keepalive = 5

# Debugging
reload = False  # Don't reload in production
spew = False  # Don't spew debug info

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = "/tmp"

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "fracture_detection_api"

# SSL (not used)
ssl_version = None
certfile = None
keyfile = None
ca_certs = None

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Server hooks
def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    pass

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")