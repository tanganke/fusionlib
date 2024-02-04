import os
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class HydraProgram:
    def __init__(self, cfg: Optional[DictConfig]):
        self.cfg = cfg
        self._output_dir = None

    @property
    def working_dir(self):
        return os.getcwd()

    @property
    def output_dir(self):
        if self._output_dir is None:
            self._output_dir = HydraConfig.get().runtime.output_dir
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value


@hydra.main()
def main(cfg: DictConfig):
    program = HydraProgram(cfg)
    print(f"Working directory: {program.working_dir}")
    print(f"Output directory: {program.output_dir}")


if __name__ == "__main__":
    main()
