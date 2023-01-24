from setuptools import setup

install_requires = [
    "numpy",
    "habitat-sim",
]

setup(
    name="home_robot_sim",
    version="0.1.0",
    packages=["home_robot_sim"],
    package_dir={"": "."},
    install_requires=install_requires,
    tests_require=["pytest"],
    zip_safe=False,
)
