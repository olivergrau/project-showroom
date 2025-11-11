| **Task**                                   | **Command**                             | **Explanation**                                                          |
| ------------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------ |
| **List all environments**                  | `conda env list` or `conda info --envs` | Shows all available environments and highlights the active one with `*`. |
| **Create new environment**                 | `conda create -n myenv python=3.10`     | Creates a new environment named `myenv` with Python 3.10.                |
| **Activate an environment**                | `conda activate myenv`                  | Switches your shell to the specified environment.                        |
| **Deactivate current environment**         | `conda deactivate`                      | Returns to the base environment.                                         |
| **List installed packages in current env** | `conda list`                            | Shows all packages installed in the active environment.                  |
| **Remove an environment**                  | `conda remove -n myenv --all`           | Deletes the environment completely.                                      |
| **Export environment to file**             | `conda env export > environment.yml`    | Saves package specs for sharing or reproduction.                         |
| **Recreate env from file**                 | `conda env create -f environment.yml`   | Builds an environment from a YAML file.                                  |
