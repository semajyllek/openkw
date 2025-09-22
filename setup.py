from setuptools import setup, find_packages

setup(
    name='openkw',

    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # use the dependencies from requirements.txt
        'torch', 
        'torchaudio', 
        'numpy', 
        'sounddevice',
        'pytorch-metric-learning',
    ],
    # Include the trained model weights and default commands.json
    package_data={
        '': ['models/kw_tcresnet_embedder.pth', 'commands.json'],
    },
    include_package_data=True,
    # Define CLI entry point for easy use
    entry_points={
        'console_scripts': [
            'openkw = deployment.spotter:cli_main',
        ],
    },
)
