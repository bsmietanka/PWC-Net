from setuptools import setup, find_packages


setup(
    name="flow_fusion",
    version="1.0",
    packages=find_packages(),
    package_data={'flow_fusion': ['checkpoints/fusion_net.pth.tar']}
)
