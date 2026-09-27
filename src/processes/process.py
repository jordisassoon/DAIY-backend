from pprint import pp

from libs.utils import fix_random_seed
from tools.print_helper import create_process


class Process:
    def __init__(self) -> None:
        if not hasattr(self, "name"):
            self.name = "_nameless_process_"
        create_process(process_info=f"You created a {self.name} process")
        self.rng_generator = fix_random_seed(0, include_cuda=True)

    def run(self) -> None:
        raise NotImplementedError

    def print_class(self) -> None:
        print("Class variables and their values:")
        class_variables = vars(self)
        pp(class_variables)
