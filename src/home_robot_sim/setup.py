from setuptools import setup

install_requires = [
    "numpy",
    # "habitat-sim",
    "home-robot",
]

setup(
    name="home-robot-sim",
    version="0.1.0",
    packages=["home_robot_sim"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
