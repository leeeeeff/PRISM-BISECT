"""Root-level setup.py — installs prism_app as a proper package.

Run from DIFFUSE/:
    pip install -e .

This makes `import prism_app` work anywhere (Streamlit, CLI, tests).
"""
from setuptools import setup, find_packages

setup(
    name='prism-app',
    version='0.1.0',
    description='PRISM+BISECT Interactive Isoform Function Analysis Web Tool',
    author='Seungwon Lee',
    packages=find_packages(include=['prism_app', 'prism_app.*']),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24',
        'pandas>=2.0',
        'scikit-learn>=1.3',
        'scipy>=1.10',
        'streamlit>=1.32',
        'plotly>=5.18',
        'umap-learn>=0.5',
        'jinja2>=3.1',
    ],
    extras_require={
        'dev': ['pytest', 'black', 'ruff'],
    },
    entry_points={
        'console_scripts': [
            'prism-app=prism_app.cli.run_analysis:main',
            'prism-app-report=prism_app.cli.generate_report:main',
        ],
    },
    package_data={
        'prism_app': [
            'data/demo/*.npy',
            'data/demo/*.json',
            'data/demo/*.tsv',
            'data/annotations/*.txt',
            'data/annotations/*.json',
        ],
    },
)
