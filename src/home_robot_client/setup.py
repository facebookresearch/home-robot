from setuptools import setup

install_requires = [
    "numpy",
    "Pyro5",
    "pynput",
]

setup(
    name="home-robot-client",
    version="0.1.0",
    packages=["home_robot_client"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
