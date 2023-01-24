from setuptools import setup

install_requires = [
    "numpy",
    "Pyro5",
    "pynput",
]

setup(
    name="home_robot_client",
    version="1.0.0",
    packages=["home_robot_client"],
    package_dir={"": "."},
    install_requires=install_requires,
    tests_require=["pytest"],
    zip_safe=False,
)
