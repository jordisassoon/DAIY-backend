from tools import bcolors


def create_process(process_info):
    print(f"{bcolors.HEADER}")
    print(process_info + f"{bcolors.ENDC}")


def run_process(process_info):
    print(f"{bcolors.OKCYAN}\n===============>")
    print(process_info)
    print(f"===============>\n{bcolors.ENDC}")


def process_output(info):
    print(process_output_builder(info))


def process_output_builder(info):
    return f"{bcolors.OKGREEN}" + info + f"{bcolors.ENDC}"
