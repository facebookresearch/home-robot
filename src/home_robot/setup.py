from setuptools import setup

install_requires = [
    "numpy",
]

# For data_tools sub-module
data_tools_requires = ["h5py", "imageio", "pygifsicle"]
install_requires += data_tools_requires

setup(
    name="home-robot",
    version="0.1.0",
    packages=["home_robot"],
    package_dir={"": "."},
    install_requires=install_requires,
    zip_safe=False,
)
