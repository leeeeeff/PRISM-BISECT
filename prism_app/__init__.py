"""PRISM Interactive Analysis Tool."""
import os as _os
# Must be set before protobuf or umap-learn is imported.
# umap-learn's parametric_umap imports TensorFlow which locks protobuf into
# C-extension mode; this forces the pure-Python fallback instead.
_os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

__version__ = '0.1.0'
