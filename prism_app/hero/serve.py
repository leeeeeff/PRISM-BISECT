"""Serve the PRISM hero landing page on port 8500.

Usage:
    python prism_app/hero/serve.py

Open:  http://localhost:8500
"""
import http.server
import socketserver
import os
import sys

PORT = 8500
HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def log_message(self, fmt, *args):
        # Suppress the verbose access log; show only errors.
        if int(args[1]) >= 400:
            super().log_message(fmt, *args)


if __name__ == '__main__':
    os.chdir(HERE)
    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f'Hero page  →  http://localhost:{PORT}')
        print(f'Streamlit  →  http://localhost:8501  (start separately)')
        print('Ctrl-C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')
            sys.exit(0)
